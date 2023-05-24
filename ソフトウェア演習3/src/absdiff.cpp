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

  // サイズの確認
  if (image1.size() != image2.size())
  {
    std::cout << "画像サイズが異なります" << std::endl;
    return -1;
  }

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

  // 抽出した赤色部分を表示
  cv::Mat redImage;
  resultImage.copyTo(redImage, redMask);

  // 結果を表示
  cv::imshow("Red Extraction", redImage);
  cv::waitKey(0);

  return 0;
}