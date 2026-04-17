// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "retinaface.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"
#include <opencv2/opencv.hpp>
/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv) {

    if (argc < 4) {
        printf("Usage: %s <model_path> <camera_index> <target_ipv4>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    std::string camera_index = argv[2];
    std::string ipv4 = argv[3];

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_retinaface_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_retinaface_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    retinaface_result result;

    // ====================== GStreamer 采集 Pipeline (摄像头) ======================
    cv::VideoCapture cap;
    std::string pipeline1 =
        "v4l2src device=" + camera_index + " io-mode=2 ! "
        "video/x-raw,format=NV12,width=1920,height=1080 ! "
        "videoconvert !"
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
        
        memset(&result, 0, sizeof(result));
        ret = inference_retinaface_model(&rknn_app_ctx, &src_image, &result);
        if (ret != 0) {
            printf("inference fail! ret=%d\n", ret);
            break;
        }
        
        // 绘制结果
        for (int i = 0; i < result.count; ++i) {
            int rx = result.object[i].box.left;
            int ry = result.object[i].box.top;
            int rw = result.object[i].box.right - result.object[i].box.left;
            int rh = result.object[i].box.bottom - result.object[i].box.top;
            draw_rectangle(&src_image, rx, ry, rw, rh, COLOR_GREEN, 3);
            char score_text[20];
            snprintf(score_text, 20, "%0.2f", result.object[i].score);
            printf("face @(%d %d %d %d) score=%f\n", result.object[i].box.left, result.object[i].box.top,
                   result.object[i].box.right, result.object[i].box.bottom, result.object[i].score);
            draw_text(&src_image, score_text, rx, ry, COLOR_RED, 20);
            for(int j = 0; j < 5; j++) {
                draw_circle(&src_image, result.object[i].ponit[j].x, result.object[i].ponit[j].y, 2, COLOR_ORANGE, 4);
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
    ret = release_retinaface_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_retinaface_model fail! ret=%d\n", ret);        
    }
    cap.release();
    writer.release();

    return 0;
}
