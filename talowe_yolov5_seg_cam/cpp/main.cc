
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "yolov5_seg.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <opencv2/opencv.hpp>


/*-------------------------------------------
                  Main Function
-------------------------------------------*/

int main(int argc, char **argv)
{

    if (argc < 4)
    {
        printf("%s <model_path> [camera_index] [ipv4]\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    std::string camera_index = argv[2];
    std::string ipv4 = argv[3];
    

    unsigned char class_colors[][3] = {
        {255, 56, 56},   // 'FF3838'
        {255, 157, 151}, // 'FF9D97'
        {255, 112, 31},  // 'FF701F'
        {255, 178, 29},  // 'FFB21D'
        {207, 210, 49},  // 'CFD231'
        {72, 249, 10},   // '48F90A'
        {146, 204, 23},  // '92CC17'
        {61, 219, 134},  // '3DDB86'
        {26, 147, 52},   // '1A9334'
        {0, 212, 187},   // '00D4BB'
        {44, 153, 168},  // '2C99A8'
        {0, 194, 255},   // '00C2FF'
        {52, 69, 147},   // '344593'
        {100, 115, 255}, // '6473FF'
        {0, 24, 236},    // '0018EC'
        {132, 56, 255},  // '8438FF'
        {82, 0, 133},    // '520085'
        {203, 56, 255},  // 'CB38FF'
        {255, 149, 200}, // 'FF95C8'
        {255, 55, 199}   // 'FF37C7'
    };

    int ret;

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    ret = init_yolov5_seg_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov5_seg_model fail! ret=%d model_path=%s\n", ret, model_path);
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    object_detect_result_list od_results;

// ====================== GStreamer pipeline ======================
    cv::VideoCapture cap;
    bool camera_opened = false;

    std::string pipeline1 = 
        "v4l2src device=" + camera_index + " io-mode=2 ! "
        "video/x-raw,format=NV12,width=1920,height=1080 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=2";

    cap.open(pipeline1, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Camera open failed!" << std::endl;
    }

// ====================== VideoWriter ======================
    cv::VideoWriter writer;
    std::string send_pipeline =
        "appsrc is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width=1920,height=1080,framerate=30/1 ! "
        "queue ! "
        "videoconvert ! "
        "video/x-raw,format=NV12 ! "
        "mpph264enc ! "
        "h264parse ! "
        "rtph264pay pt=96 ! "
        "udpsink host=" + ipv4 + " port=5000 sync=false async=false";

    writer.open(send_pipeline, cv::CAP_GSTREAMER, 0, 30, cv::Size(1920, 1080), true);
    // writer.open(send_pipeline, cv::CAP_GSTREAMER, 0, 30, cv::Size(3840, 2160), true);

    if (!writer.isOpened()) {
        std::cerr << "Failed to open VideoWriter for network streaming!" << std::endl;
        std::cerr << "Pipeline: " << send_pipeline << std::endl;
        return -1;
    }
    std::cout << "VideoWriter successfully!" << std::endl;

    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    double fps = 0.0;

    double recent_npu_times[10] = {0}; 
    int npu_time_index = 0; 
    int npu_time_count = 0; 
    double avg_npu_time = 0.0; 
    double npu_fps = 0.0; 

    cv::Mat src_frame;
    while(true){
        auto frame_start = std::chrono::steady_clock::now();
        cap >> src_frame;
        if (src_frame.empty()) {
            std::cerr << "Failed to grab frame from the camera." << std::endl;
            break;
        }

        // cv::rotate(src_frame, src_frame, cv::ROTATE_90_CLOCKWISE);
        src_image.height = src_frame.rows;
        src_image.width = src_frame.cols;
        src_image.width_stride = src_frame.step[0];
        src_image.virt_addr = src_frame.data;
        src_image.format = IMAGE_FORMAT_RGB888;
        src_image.size = src_frame.total() * src_frame.elemSize();

        int ret;
        image_buffer_t dst_img;
        dst_img.virt_addr = NULL;
        letterbox_t letter_box;
        rknn_input inputs[rknn_app_ctx.io_num.n_input];
        rknn_output outputs[rknn_app_ctx.io_num.n_output];
        const float nms_threshold = NMS_THRESH;
        const float box_conf_threshold = BOX_THRESH;
        int bg_color = 114; // pad color for letterbox

        memset(&od_results, 0x00, sizeof(od_results));
        memset(&letter_box, 0, sizeof(letterbox_t));
        memset(&dst_img, 0, sizeof(image_buffer_t));
        memset(inputs, 0, sizeof(inputs));
        memset(outputs, 0, sizeof(outputs));

        // Pre Process
        rknn_app_ctx.input_image_width = src_image.width;
        rknn_app_ctx.input_image_height = src_image.height;
        dst_img.width = rknn_app_ctx.model_width;
        dst_img.height = rknn_app_ctx.model_height;
        dst_img.format = IMAGE_FORMAT_RGB888;
        dst_img.size = get_image_size(&dst_img);
        dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
        if (dst_img.virt_addr == NULL)
        {
            printf("malloc buffer size:%d fail!\n", dst_img.size);
            return -1;
        }

        // letterbox
        ret = convert_image_with_letterbox(&src_image, &dst_img, &letter_box, bg_color);
        if (ret < 0)
        {
            printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        }

        // Set Input Data
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].size = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
        inputs[0].buf = dst_img.virt_addr;

        ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
        }
        free(dst_img.virt_addr);

        // Run
        auto npu_start = std::chrono::steady_clock::now();
        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);

        }

        auto npu_end = std::chrono::steady_clock::now();
        double current_npu_time = std::chrono::duration<double, std::milli>(npu_end - npu_start).count();
        
        recent_npu_times[npu_time_index] = current_npu_time;
        npu_time_index = (npu_time_index + 1) % 10;
        if (npu_time_count < 10) {
            npu_time_count++;
        }

        double total_npu_time = 0.0;
        for (int i = 0; i < npu_time_count; i++) {
            total_npu_time += recent_npu_times[i];
        }
        avg_npu_time = total_npu_time / npu_time_count;
        npu_fps = avg_npu_time > 0 ? 1000.0 / avg_npu_time : 0.0;

        // Get Output
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < rknn_app_ctx.io_num.n_output; i++)
        {
            outputs[i].index = i;
            outputs[i].want_float = (!rknn_app_ctx.is_quant);
        }

        ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);
        if (ret < 0)
        {
            printf("rknn_outputs_get fail! ret=%d\n", ret);

        }

        // Post Process
        post_process(&rknn_app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, &od_results);

        // Remeber to release rknn output
        rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);

        // draw mask
        if (od_results.count >= 1)
        {
            int width = src_image.width;
            int height = src_image.height;
            char *ori_img = (char *)src_image.virt_addr;
            int cls_id = od_results.results[0].cls_id;
            uint8_t *seg_mask = od_results.results_seg[0].seg_mask;

            float alpha = 0.5f; // opacity
            for (int j = 0; j < height; j++)
            {
                for (int k = 0; k < width; k++)
                {
                    int pixel_offset = 3 * (j * width + k);
                    if (seg_mask[j * width + k] != 0)
                    {
                        ori_img[pixel_offset + 0] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][0] * (1 - alpha) + ori_img[pixel_offset + 0] * alpha, 0, 255); // r
                        ori_img[pixel_offset + 1] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][1] * (1 - alpha) + ori_img[pixel_offset + 1] * alpha, 0, 255); // g
                        ori_img[pixel_offset + 2] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][2] * (1 - alpha) + ori_img[pixel_offset + 2] * alpha, 0, 255); // b
                    }
                }
            }
            free(seg_mask);
        }

        // draw boxes
        char text[256];
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            printf("[DET] %-9s | box:(%4d ,%4d)(%4d ,%4d) | size: %3d*%3d | conf:%6.2f%%\n",
                coco_cls_to_name(det_result->cls_id),
                det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                det_result->box.right - det_result->box.left, det_result->box.bottom - det_result->box.top,
                det_result->prop*100);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_RED, 3);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 16, COLOR_BLUE, 10);
        }

        frame_count++;
        auto frame_end = std::chrono::steady_clock::now();
        double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
        
        if (frame_count % 10 == 0) { 
            auto current_time = std::chrono::steady_clock::now();
            double elapsed_time = std::chrono::duration<double>(current_time - start_time).count();
            fps = frame_count / elapsed_time;
        }
        
        char fps_text[128];
        sprintf(fps_text, "camera FPS:%.1f\nnpu FPS:%.1f", fps, npu_fps);
        draw_text(&src_image, fps_text, 20, 20, COLOR_GREEN, 12);
    
        writer.write(src_frame);
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    double avg_fps = frame_count / total_elapsed_time;
    printf("Average FPS: %.2f over %d frames (%.2f seconds)\n", avg_fps, frame_count, total_elapsed_time);

    deinit_post_process();

    ret = release_yolov5_seg_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_yolov5_seg_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }
    return 0;
}
