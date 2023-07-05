#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char **argv)
{
  // 元の画像と間違いが含まれた画像の読み込み
  cv::Mat src = cv::imread(argv[1]);
  cv::Mat dst = cv::imread(argv[2]);

  // 読み込みの確認
  if (src.empty() || dst.empty())
  {
    std::cout << "読み込みに失敗しました" << std::endl;
    return -1;
  }

  // 画像のリサイズ
  cv::resize(dst, dst, src.size());

  // 画像の差分を計算
  cv::Mat diffImage;
  cv::absdiff(src, dst, diffImage);

  // 差分画像をグレースケールに変換
  cv::cvtColor(diffImage, diffImage, cv::COLOR_BGR2GRAY);

  // 差分画像を2値化
  cv::threshold(diffImage, diffImage, 30, 255, cv::THRESH_BINARY);

  // 差分のある画素に赤色を付ける
  cv::Mat resultImage = src.clone();
  resultImage.setTo(cv::Scalar(0, 0, 255), diffImage);

  cv::Scalar lowerRed = cv::Scalar(0, 0, 255);
  cv::Scalar upperRed = cv::Scalar(0, 0, 255);

  // 画像から赤色部分を抽出
  cv::Mat redMask;
  cv::inRange(resultImage, lowerRed, upperRed, redMask);

  cv::Mat redImage;
  resultImage.copyTo(redImage, redMask);

  // オープニング（収縮と膨張）を適用
  int iterations = 1; // 収縮と膨張の回数

  // 膨張処理を行うカーネルの定義
  cv::Mat kernel;
  cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

  cv::morphologyEx(redImage, redImage, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), iterations);

  // redImageの輪郭を計算します。
  cv::Mat redGray;
  cv::cvtColor(redImage, redGray, cv::COLOR_BGR2GRAY);
  cv::blur(redGray, redGray, cv::Size(3, 3));
  cv::Mat redCanny;
  int thresh = 150;
  cv::Canny(redGray, redCanny, thresh, thresh * 2);
  std::vector<std::vector<cv::Point>> Contours;
  cv::findContours(redCanny, Contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  std::cout << Contours.size() << std::endl;

  // redImageの輪郭を描画します。
  cv::Mat dstRedContours = cv::Mat::zeros(redCanny.size(), CV_8UC3);
  for (size_t i = 0; i < Contours.size(); i++)
  {
    cv::Scalar color = cv::Scalar(0, 0, 255); // 赤色の輪郭を描画するためにScalar(0, 0, 255)を使用します
    cv::drawContours(dstRedContours, Contours, (int)i, color);
  }

  // 少ない点で輪郭を近似します。
  std::vector<std::vector<cv::Point>> Contours_poly(Contours.size());
  for (size_t i = 0; i < Contours.size(); i++)
  {
    std::vector<cv::Point> poly;
    cv::approxPolyDP(Contours[i], poly, 3, true);
    Contours_poly[i] = poly;
  }

  cv::Mat dstContoursPoly = cv::Mat::zeros(redCanny.size(), CV_8UC3);
  for (size_t i = 0; i < Contours.size(); i++)
  {
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::drawContours(dstContoursPoly, Contours_poly, (int)i, color);
  }

    // x-y 軸に沿ったバウンディングボックス
    std::vector<cv::Rect> boundRect(Contours.size());

    // 最小内包円
    std::vector<cv::Point2f>centers(Contours.size());
    std::vector<float>radius(Contours.size());

    for(size_t i = 0; i < Contours.size(); i++) {
        boundRect[i] = cv::boundingRect(Contours_poly[i]);
        cv::minEnclosingCircle(Contours_poly[i], centers[i], radius[i]);
    }

    cv::Mat dst2 = redImage.clone();
    for( size_t i = 0; i< Contours.size(); i++ ) {
        cv::Scalar color = cv::Scalar(0, 255, 255);
       // cv::rectangle(dst2, boundRect[i].tl(), boundRect[i].br(), color, 2);
        cv::circle(dst2, centers[i], (int)radius[i], color, 2);
    }
 
  // 画像の暗くする度合いを設定
  double alpha = 0.4; // 0.0から1.0の範囲で設定（1.0で元の明るさ、0.0で完全に暗くなる）

  // 画像を暗くする
  cv::Mat darkenedImage = src * alpha;

  // cleanedRedImageと暗くした画像を重ねる
  cv::Mat overlaidImage;
  cv::addWeighted(dst2, 1.0, darkenedImage, 1.0, 0.0, overlaidImage);

  // 結果を表示

  cv::imshow("Red Extraction", overlaidImage);
  cv::waitKey(0);

  return 0;
}