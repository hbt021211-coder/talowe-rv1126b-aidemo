#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>

#include "yolov8-pose.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include <opencv2/opencv.hpp>

// 骨架连接定义
int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8,
                    7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printf("Usage: %s <model_path> <camera_index> <target_ipv4>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    std::string camera_index = argv[2];
    std::string ipv4 = argv[3];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    init_post_process();

    // 初始化 YOLOv8-Pose 模型
    ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_yolov8_pose_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    object_detect_result_list od_results;

    // ====================== GStreamer 采集 Pipeline (摄像头) ======================
    cv::VideoCapture cap;
    std::string pipeline1 =
        "v4l2src device=" + camera_index + " io-mode=2 ! "
        "video/x-raw,format=NV12,width=1920,height=1080 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=2";

    cap.open(pipeline1, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Camera open failed!" << std::endl;
        return -1;
    }

    // ====================== GStreamer 推流 Pipeline (UDP) ======================
    cv::VideoWriter writer;
    std::string send_pipeline =
        "appsrc is-live=true block=true format=time "
        "caps=video/x-raw,format=BGR,width=1920,height=1080,framerate=30/1 ! "
        "queue ! videoconvert ! video/x-raw,format=NV12 ! "
        "mpph264enc ! h264parse ! rtph264pay pt=96 ! "
        "udpsink host=" + ipv4 + " port=5000 sync=false async=false";

    writer.open(send_pipeline, cv::CAP_GSTREAMER, 0, 30, cv::Size(1920, 1080), true);
    if (!writer.isOpened()) {
        std::cerr << "Failed to open VideoWriter!" << std::endl;
        return -1;
    }

    std::cout << "Pipeline initialized. Starting inference..." << std::endl;

    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    double fps = 0.0;
    cv::Mat frame;

    while (true)
    {
        auto frame_start = std::chrono::steady_clock::now();
        cap >> frame;
        if (frame.empty()) break;

        // 推理所需的 image_buffer_t
        src_image.height = frame.rows;
        src_image.width = frame.cols;
        src_image.width_stride = frame.step[0];
        src_image.virt_addr = frame.data;
        src_image.format = IMAGE_FORMAT_RGB888; 
        src_image.size = frame.total() * frame.elemSize();

        // 执行推理
        memset(&od_results, 0, sizeof(od_results));
        ret = inference_yolov8_pose_model(&rknn_app_ctx, &src_image, &od_results);
        if (ret != 0) {
            printf("inference fail! ret=%d\n", ret);
            break;
        }

        // 绘制结果
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result *det_result = &(od_results.results[i]);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;

            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

            char text[256];
            sprintf(text, "person %.1f%%", det_result->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);

            for (int j = 0; j < 38 / 2; ++j)
            {
                int k1 = skeleton[2 * j] - 1;
                int k2 = skeleton[2 * j + 1] - 1;
                draw_line(&src_image, 
                          (int)det_result->keypoints[k1][0], (int)det_result->keypoints[k1][1],
                          (int)det_result->keypoints[k2][0], (int)det_result->keypoints[k2][1], 
                          COLOR_ORANGE, 3);
            }

            for (int j = 0; j < 17; ++j)
            {
                draw_circle(&src_image, (int)det_result->keypoints[j][0], (int)det_result->keypoints[j][1], 2, COLOR_YELLOW, -1);
            }
        }

        // 统计 FPS 并叠加到图像
        frame_count++;
        if (frame_count % 10 == 0) {
            auto current_time = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(current_time - start_time).count();
            fps = frame_count / elapsed;
        }
        char fps_text[64];
        sprintf(fps_text, "FPS: %.1f", fps);
        draw_text(&src_image, fps_text, 50, 50, COLOR_GREEN, 15);

        // 推流
        writer.write(frame);
    }

    // 释放资源
    deinit_post_process();
    release_yolov8_pose_model(&rknn_app_ctx);
    cap.release();
    writer.release();

    return 0;
}