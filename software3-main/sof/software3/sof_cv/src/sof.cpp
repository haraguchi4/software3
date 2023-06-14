#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <numeric>

//比較する２つの画像のフルパスを格納するグローバル変数
std::string g_img_path1, g_img_path2;

/*  特徴点マッチングによる類似度計算及び入力画像の補正(射影変換)を行う関数 
    src1 : 入力画像1
    src2 : 入力画像2
    dst1 : 出力画像1
    dst2 : 出力画像2
    
    動作確認済
    1. 特徴点検出:ORB   + 特徴点マッチング:総当たり(BMF)
    2. 特徴点検出:AKAZE + 特徴点マッチング:総当たり(BMF)
    3. 特徴点検出:ORB   + 特徴点マッチング:FLANN + ratio test
    4. 特徴点検出:AKAZE + 特徴点マッチング:FLANN + ratio test
    
    必要処理時間: 3 < 1 < 4 < 2 
    精度　　　　: 1 = 3 < 2 = 4    */

void feature_matching(const cv::Mat &src1, const cv::Mat &src2, cv::Mat &dst1, cv::Mat &dst2)
{
  std::vector<cv::KeyPoint> key1, key2; // 特徴点を格納
  cv::Mat des1, des2; // 特徴量記述の計算

  /* 2画像を比較して2辺とも小さい場合はsrc2に合わせて補正できるように */
  if(src1.cols <= src2.cols && src1.rows <= src2.rows){
    dst1 = src2;
    dst2 = src1;
  }else{
    dst1 = src1;
    dst2 = src2;
  }

  /* 比較のために複数手法を記述 必要に応じてコメントアウト*/
  /* 特徴点検出*/
  /* AKAZE */
  cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
  akaze->detectAndCompute(dst1, cv::noArray(), key1, des1);
  akaze->detectAndCompute(dst2, cv::noArray(), key2, des2);
  /* ORB */
  // cv::Ptr<cv::ORB> orb = cv::ORB::create();
  // orb->detectAndCompute(dst1, cv::noArray(), key1, des1);
  // orb->detectAndCompute(dst2, cv::noArray(), key2, des2);

  //std::cout << des1 << std::endl;

  /* 特徴点マッチングアルゴリズム */
  cv::Ptr<cv::DescriptorMatcher> matcher;
  std::vector<cv::DMatch> match;
  /* 総当たり */
  // matcher = cv::DescriptorMatcher::create("BruteForce");
  // matcher->match(des1, des2, match);
  /* FLANN */
  matcher = cv::DescriptorMatcher::create("FlannBased");
  std::vector<std::vector<cv::DMatch>> knn_matches;
  if(des1.type() != CV_32F) {
    des1.convertTo(des1, CV_32F);
  }
  if(des2.type() != CV_32F) {
    des2.convertTo(des2, CV_32F);
  }
  matcher->knnMatch(des1, des2, knn_matches, 2);
  /* ratio test */
  const float RATIO_THRESH = 0.7f;
  for(int i = 0; i < knn_matches.size(); i++){
    if(knn_matches[i][0].distance < RATIO_THRESH * knn_matches[i][1].distance){
      match.push_back(knn_matches[i][0]);
    }
  }

  /* 類似度計算(距離による実装、0に近い値ほど画像が類似) */
  float sim = 0;
  const float THRESHOLD = 300; // 類似度の閾値(仮)
  for(int i = 0; i < match.size(); i++){
    cv::DMatch dis = match[i];
    sim += dis.distance;
  }
  sim /= match.size();
  std::cout << "類似度: " << sim << std::endl; 

  /* 画像の類似度が低すぎる場合は終了 */
  if(0/* sim > THRESHOLD*/){
    std::cerr << "画像が違いすぎます" << std::endl;
    std::exit(1);
  }

  /* 特徴量距離の小さい順にソートし、不要な点を削除 */
  for(int i = 0; i < match.size(); i++){
    double min = match[i].distance;
    int n = i;
    for(int j = i + 1; j < match.size(); j++){
      if(min > match[j].distance){
        n = j;
        min = match[j].distance;
      }
    }
    std::swap(match[i], match[n]);
  }
  if(match.size() > 50){
    match.erase(match.begin() + 50, match.end());
  }else{
    match.erase(match.begin() + match.size(), match.end());
  }

  //cv::drawMatches(src1, key1, src2, key2, match, dst);

  /* src1をsrc2に合わせる形で射影変換して補正 */
  std::vector<cv::Vec2f> get_pt1(match.size()), get_pt2(match.size()); // 使用する特徴点
  /* 対応する特徴点の座標を取得・格納*/
  for(int i = 0; i < match.size(); i++){
    get_pt1[i][0] = key1[match[i].queryIdx].pt.x;
    get_pt1[i][1] = key1[match[i].queryIdx].pt.y;
    get_pt2[i][0] = key2[match[i].trainIdx].pt.x;
    get_pt2[i][1] = key2[match[i].trainIdx].pt.y;
  }

  /* ホモグラフィ行列推定 */
  cv::Mat H = cv::findHomography(get_pt1, get_pt2, cv::RANSAC); 
  /* src1を変形 */
  cv::warpPerspective(dst1, dst1, H, dst2.size());
}

void absdiff1(const cv::Mat &src, const cv::Mat &dst)
{
  // 読み込みの確認
  if (src.empty() || dst.empty())
  {
    std::cout << "読み込みに失敗しました" << std::endl;
    return;
  }

  // 画像のリサイズ
  cv::resize(dst, dst, src.size());

  // 画像の差分を計算
  cv::Mat diffImage;
  cv::absdiff(src, dst, diffImage);

  // 差分画像をグレースケールに変換
  cv::cvtColor(diffImage, diffImage, cv::COLOR_BGR2GRAY);

  // ブラーでノイズ除去
  cv::medianBlur(diffImage, diffImage, 3);

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

  // 画像の暗くする度合いを設定
  double alpha = 0.3; // 0.0から1.0の範囲で設定（1.0で元の明るさ、0.0で完全に暗くなる）

  // 画像を暗くする
  cv::Mat darkenedImage = src * alpha;

  // redMaskと暗くした画像を重ねる
  cv::Mat overlaidImage;
  cv::addWeighted(redImage, 1.0, darkenedImage, 1.0, 0.0, overlaidImage);

  cv::resize(overlaidImage, overlaidImage, cv::Size(), 500.0/overlaidImage.cols ,500.0/overlaidImage.cols);
  
  cv::imshow("Resuit", overlaidImage);
  cv::waitKey(0);

  cv::imwrite("Result.png", overlaidImage);
}

void absdiff2(const cv::Mat &src, const cv::Mat &dst)
{
  // 読み込みの確認
  if (src.empty() || dst.empty())
  {
    std::cout << "読み込みに失敗しました" << std::endl;
    return;
  }

  // 画像のリサイズ
  cv::resize(dst, dst, src.size());

  cv::Mat WarpedSrcMat, LmatFloat, Rsrc;
  WarpedSrcMat.convertTo(LmatFloat, CV_16SC3);
  Rsrc.convertTo(Rsrc, CV_16SC3);

  std::vector<cv::Mat> planes1;
  std::vector<cv::Mat> planes2;

  cv::Mat diff0;
  cv::Mat diff1;
  cv::Mat diff2;

	// 3つのチャネルB, G, Rに分離 (OpenCVではデフォルトでB, G, Rの順)
	cv::split(src, planes1);
  cv::split(dst, planes2);

  ///分割したチャンネルごとに差分を出す
  cv::absdiff(planes1[0], planes2[0], diff0);
  cv::absdiff(planes1[1], planes2[1], diff1);
  cv::absdiff(planes1[2], planes2[2], diff2);

  // ブラーでノイズ除去
  cv::medianBlur(diff0, diff0, 3);
  cv::medianBlur(diff1, diff1, 3);
  cv::medianBlur(diff2, diff2, 3);

  cv::Mat wiseMat;
  cv::bitwise_or(diff0, diff1, wiseMat);
  cv::bitwise_or(wiseMat, diff2, wiseMat);
  
  //オープニング処理でノイズ緩和
  cv::Mat openingMat;
  cv::morphologyEx(wiseMat, openingMat, 0, openingMat);

  // スレッショルドで差分をきれいにくっきりと
  cv::Mat dilationMat;
  cv::dilate(openingMat, dilationMat, dilationMat);
  cv::threshold(dilationMat, dilationMat, 100, 255, cv::THRESH_BINARY);
  cv::medianBlur(dilationMat, dilationMat, 3);

  // dilationMatはグレースケールなので合成先のMatと同じ色空間に変換する
  cv::Mat dilationScaleMat;
  cv::Mat dilationColorMat;
  cv::convertScaleAbs(dilationMat, dilationScaleMat);
  cv::cvtColor(dilationScaleMat, dilationColorMat, cv::COLOR_GRAY2RGB);
 
  // 元画像 3:差分画像 7 で合成
  cv:: Mat Result;
  cv::addWeighted(src, 0.3, dilationColorMat, 0.7, 0, Result);
  //cv::addWeighted(, 0.3, dilationColorMat, 0.7, 0, RaddMat);

  resize(Result, Result, cv::Size(), 500.0/Result.cols , 500.0/Result.cols);

  cv::imshow("result", Result);
  cv::waitKey(0);
}

int main(int argc, char **argv)
{
  // 元の画像と間違いが含まれた画像の読み込み
  cv::Mat image1 = cv::imread(argv[1]);
  cv::Mat image2 = cv::imread(argv[2]);

  
  //特徴点マッチング
  feature_matching(image1, image2, image1, image2);
  
  int num = atoi(argv[3]);
 
  if(num == 1){
    absdiff1(image1, image2); 
  }
  else {
    absdiff2(image1, image2); 
  }

  return 0;
}