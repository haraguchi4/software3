#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char **argv)
{
  // 元の画像と間違いが含まれた画像の読み込み
  cv::Mat image1 = cv::imread(argv[1]);
  cv::Mat image2 = cv::imread(argv[2]);

  // 読み込みの確認
  if (image1.empty() || image2.empty())
  {
    std::cout << "読み込みに失敗しました" << std::endl;
    return -1;
  }

  // 画像のリサイズ
  cv::resize(image2, image2, image1.size());

  // 画像の差分を計算
  cv::Mat diffImage;
  cv::absdiff(image1, image2, diffImage);

  // 差分画像をグレースケールに変換
  cv::cvtColor(diffImage, diffImage, cv::COLOR_BGR2GRAY);

  // 差分画像を2値化
  cv::threshold(diffImage, diffImage, 30, 255, cv::THRESH_BINARY);

  // 差分のある画素に赤色を付ける
  cv::Mat resultImage = image1.clone();
  resultImage.setTo(cv::Scalar(0, 0, 255), diffImage);

  cv::Scalar lowerRed = cv::Scalar(0, 0, 255);
  cv::Scalar upperRed = cv::Scalar(0, 0, 255);

  // 画像から赤色部分を抽出
  cv::Mat redMask;
  cv::inRange(resultImage, lowerRed, upperRed, redMask);

  cv::Mat redImage;
  resultImage.copyTo(redImage, redMask);

  // 画像の暗くする度合いを設定
  double alpha = 0.4; // 0.0から1.0の範囲で設定（1.0で元の明るさ、0.0で完全に暗くなる）

  // 画像を暗くする
  cv::Mat darkenedImage = image1 * alpha;

  // redMaskと暗くした画像を重ねる
  cv::Mat overlaidImage;
  cv::addWeighted(redImage, 1.0, darkenedImage, 1.0, 0.0, overlaidImage);

  // 結果を表示
  cv::imshow("Red Extraction", overlaidImage);
  cv::waitKey(0);

  return 0;
}