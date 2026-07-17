#include <iostream>
#include <vector>
#include <zmq.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main() {
    // 1. ZeroMQのコンテキストとSUBソケットの作成
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::sub);

    // 2. 送信側（Python）のIPアドレスとポートを指定して接続
    std::string server_address = "tcp://192.168.123.164:5555"; 
    socket.connect(server_address);
    std::cout << "Connected to " << server_address << std::endl;

    // 3. フィルタなしですべてのメッセージを受信する設定 (重要)
    socket.set(zmq::sockopt::subscribe, "");

    // リアルタイム表示用ウィンドウ
    cv::namedWindow("Received Video", cv::WINDOW_AUTOSIZE);

    cv::Mat cameraMatrix =
        (cv::Mat_<double>(3,3) <<
            606.6430, 0, 327.9997,
            0, 606.2543, 244.7415,
            0, 0, 1);

    cv::Mat distCoeffs = cv::Mat::zeros(1,5,CV_64F);

    auto dictionary =
        cv::aruco::getPredefinedDictionary(
            cv::aruco::DICT_4X4_50);

    constexpr float markerLength = 0.17f;

    while (true) {
        zmq::message_t message;

        // 4. メッセージ（JPEGバイト列）をブロッキング受信
        auto result = socket.recv(message, zmq::recv_flags::none);
        if (!result) continue;

        // 5. 受信データを uchar の vector に変換
        std::vector<uchar> buffer(
            static_cast<uchar*>(message.data()), 
            static_cast<uchar*>(message.data()) + message.size()
        );

        // 6. バイト列からOpenCVのMat画像にデコード
        cv::Mat frame = cv::imdecode(buffer, cv::IMREAD_COLOR);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        cv::aruco::detectMarkers(
            frame,
            dictionary,
            corners,
            ids);

        if(!ids.empty())
        {
            cv::aruco::drawDetectedMarkers(
                frame,
                corners,
                ids);

            std::vector<cv::Vec3d> rvecs,tvecs;

            cv::aruco::estimatePoseSingleMarkers(
                corners,
                markerLength,
                cameraMatrix,
                distCoeffs,
                rvecs,
                tvecs);

            for(size_t i=0;i<ids.size();i++)
            {
                cv::drawFrameAxes(
                    frame,
                    cameraMatrix,
                    distCoeffs,
                    rvecs[i],
                    tvecs[i],
                    0.03);

                std::cout
                    << "ID=" << ids[i]
                    << "  x=" << tvecs[i][0]
                    << " y=" << tvecs[i][1]
                    << " z=" << tvecs[i][2]
                    << std::endl;
            }
        }

        if (!frame.empty()) {
            // 画像を表示
            cv::imshow("Received Video", frame);
        } else {
            std::cerr << "Failed to decode frame." << std::endl;
        }

        // 'q' キーが押されたらループを抜けて終了
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}

