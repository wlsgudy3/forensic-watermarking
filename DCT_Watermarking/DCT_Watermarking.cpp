#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <cmath>
#include <bitset>
#include <vector>

using namespace cv;
using namespace std;

// 함수 선언
void embedWatermark(Mat& src, const vector<int>& watermark, int strength, unsigned int seed);
vector<int> extractWatermark(Mat& src, int watermarkSize, int strength, unsigned int seed);
void applyDCT(Mat& block, Mat& dctBlock);
void applyIDCT(Mat& dctBlock, Mat& block);
double calculatePSNR(const Mat& original, const Mat& processed);
vector<int> stringToBits(const string& str);
string bitsToString(const vector<int>& bits);
void attack1(Mat& src, float stddev);
void attack2(Mat& src);
void attack3(Mat& src, double alpha);
Mat attack4(Mat& src);

// 성공한 조합 (strength, 압축율, numCopies)
// lena512.jpg : (50, 70, 3) // (50, 50, 3)은 실패
// test1.png : (30, 50, 3), (40, 50, 3), (50, 50, 3), (3, 30, 40)  // (30, 40, 3)
// test2.jpg : (20, 80, 5) (30, 80, 5), (40, 80, 5), (50, 80, 5), (60, 90, 5) // (10, 80, 5), (50, 70, 5), (60, 90, 3) 실패

int numCopies = 5;
int strength = 30; // 워터마크 강도

int compressRate = 90; // 압축 강도, 압축률이 100일 경우는 그대로 반환하게 됨.
float noise_std = 50; // noise 강도(표준편차)


// 기본적으로는 압축이나 공격이 주석처리되어 있어서 상황에 맞게 주석을 제거해줘야 함.

int main() {
    Mat image = imread("IU.jpg"); // Imgae 종류 : lena.jpg  landscape.jpg  IU.jpg
    if (image.empty()) {
        cerr << "이미지를 불러올 수 없습니다!" << endl;
        return -1;
    }

    imshow("source image", image);

    // 문자열 워터마크
    string watermarkText = "192.168.0.1 Choijinu";
    vector<int> watermarkBits = stringToBits(watermarkText);

    unsigned int seed = 42; // seed 값

    // 워터마크 삽입
    Mat watermarkedImage = image.clone();
    embedWatermark(watermarkedImage, watermarkBits, strength, seed);

    imshow("watermarked image", watermarkedImage);

    printf("PSNR : %f\n", calculatePSNR(image, watermarkedImage)); // 워터마크 삽입한 이미지와의 PSNR

    // 공격들
    //attack1(watermarkedImage, noise_std); // 가우시안 노이즈
    //attack2(watermarkedImage); // 네모 그리기
    //attack3(watermarkedImage, 0.3); // 가시성 워터마크
    
    Mat compressedImage = attack4(watermarkedImage); // 압축률이 100일경우 그대로 반환하게 되어있음.
    
    //imshow("after attack", watermarkedImage); // 공격결과 확인용 코드



    // 워터마크 추출
    vector<int> extractedBits = extractWatermark(compressedImage, watermarkBits.size(), strength, seed);

    string extractedText = bitsToString(extractedBits);

    // 결과 출력
    cout << "원본 워터마크 텍스트: " << watermarkText << endl;
    cout << "추출된 워터마크 텍스트: " << extractedText << endl;

    imshow("final Image", compressedImage);

    printf("PSNR2 : %f\n", calculatePSNR(image, compressedImage)); // 공격까지 진행한 이미지와의 PSNR

    waitKey(0);

    return 0;
}

void applyDCT(Mat& block, Mat& dctBlock) {
    int N = block.rows;
    Mat floatBlock;
    block.convertTo(floatBlock, CV_32F);
    dctBlock = Mat::zeros(N, N, CV_32F);

    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            float sum = 0.0;
            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    sum += floatBlock.at<float>(x, y) *
                        cos((2 * x + 1) * u * CV_PI / (2 * N)) *
                        cos((2 * y + 1) * v * CV_PI / (2 * N));
                }
            }
            float Cu = (u == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
            float Cv = (v == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
            dctBlock.at<float>(u, v) = Cu * Cv * sum;
        }
    }
}

void applyIDCT(Mat& dctBlock, Mat& block) {
    int N = dctBlock.rows;
    Mat idctBlock = Mat::zeros(N, N, CV_32F);

    for (int x = 0; x < N; x++) {
        for (int y = 0; y < N; y++) {
            float sum = 0.0;
            for (int u = 0; u < N; u++) {
                for (int v = 0; v < N; v++) {
                    float Cu = (u == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
                    float Cv = (v == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
                    sum += Cu * Cv * dctBlock.at<float>(u, v) *
                        cos((2 * x + 1) * u * CV_PI / (2 * N)) *
                        cos((2 * y + 1) * v * CV_PI / (2 * N));
                }
            }
            idctBlock.at<float>(x, y) = sum;
        }
    }

    idctBlock.convertTo(block, CV_8U);
}


void embedWatermark(Mat& src, const vector<int>& watermark, int strength, unsigned int seed) {
    RNG rng(seed); // 난수 생성
    Mat imageYCrCb;
    cvtColor(src, imageYCrCb, COLOR_BGR2YCrCb);

    vector<Mat> channels;
    split(imageYCrCb, channels);

    // Y 채널만 사용
    Mat& Y = channels[0];

    for (int i = 0; i < watermark.size(); ++i) {
        for (int j = 0; j < numCopies; ++j) {
            int x = rng.uniform(0, Y.rows / 8) * 8; // 8x8 블록 선택
            int y = rng.uniform(0, Y.cols / 8) * 8;

            Mat block = Y(Rect(y, x, 8, 8));
            Mat dctBlock;
            applyDCT(block, dctBlock);

            // 워터마크 삽입 (DCT 계수 중간 위치 조정)
            dctBlock.at<float>(2, 2) += strength * (watermark[i] == 1 ? 1 : -1);

            // 역 DCT 수행
            applyIDCT(dctBlock, block);
        }
    }

    merge(channels, imageYCrCb);
    cvtColor(imageYCrCb, src, COLOR_YCrCb2BGR);
}

vector<int> extractWatermark(Mat& src, int watermarkSize, int strength, unsigned int seed) {
    RNG rng(seed); // 난수 생성
    Mat imageYCrCb;
    cvtColor(src, imageYCrCb, COLOR_BGR2YCrCb);

    vector<Mat> channels;
    split(imageYCrCb, channels);

    // Y 채널만 사용
    Mat& Y = channels[0];

    vector<int> extractedWatermark(watermarkSize);

    for (int i = 0; i < watermarkSize; ++i) {
        int votes = 0; // 다수결을 위한 투표 변수
        for (int j = 0; j < numCopies; ++j) {
            int x = rng.uniform(0, Y.rows / 8) * 8; // 8x8 블록 선택
            int y = rng.uniform(0, Y.cols / 8) * 8;

            Mat block = Y(Rect(y, x, 8, 8));
            Mat dctBlock;
            applyDCT(block, dctBlock);

            // 워터마크 비트 추출 및 투표
            votes += (dctBlock.at<float>(2, 2) > 0) ? 1 : -1;
        }

        // 다수결에 따라 워터마크 비트 결정
        extractedWatermark[i] = (votes > 0) ? 1 : 0;
    }

    return extractedWatermark;
}


// PSNR 계산
double calculatePSNR(const Mat& original, const Mat& processed) {
    Mat diff;
    absdiff(original, processed, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = sum(diff)[0] / (double)(original.total() * original.channels());
    if (mse == 0) return INFINITY;

    return 10.0 * log10((255 * 255) / mse);
}

vector<int> stringToBits(const string& str) {
    vector<int> bits;
    for (char c : str) {
        bitset<8> charBits(c); // 1문자를 8비트로 변환
        for (int i = 7; i >= 0; --i) {
            bits.push_back(charBits[i]);
        }
    }
    return bits;
}

// 비트열 -> 문자열 변환
string bitsToString(const vector<int>& bits) {
    string str;
    for (size_t i = 0; i < bits.size(); i += 8) {
        bitset<8> charBits;
        for (int j = 0; j < 8; ++j) {
            charBits[7 - j] = bits[i + j];
        }
        str.push_back(static_cast<char>(charBits.to_ulong()));
    }
    return str;
}

// 가우시안 노이즈
void attack1(Mat& src, float stddev) {
    cv::Mat noise = cv::Mat(src.size(), src.type());

    // 가우시안 노이즈 생성 (평균값: mean, 표준편차: stddev)
    cv::randn(noise, 0, stddev);

    src += noise;
}

// 사각형 그림 그리기
void attack2(Mat& src) {
    cv::rectangle(src, Point(200, 200), Point(300, 300), Scalar(0, 0, 0), 5, cv::LINE_8);
}

void attack3(Mat& src, double alpha) { // alpha: 워터마크 투명도 (0: 완전히 투명, 1: 불투명)
    Mat visibleWatermark = imread("symbol1.png", IMREAD_UNCHANGED);
    if (visibleWatermark.empty()) {
        cerr << "워터마크 이미지를 불러올 수 없습니다!" << endl;
        return;
    }

    imshow("visible watermark", visibleWatermark);

    // 워터마크 크기 조정 (원본 이미지에 맞게)
    Mat resizedWatermark;
    resize(visibleWatermark, resizedWatermark, Size(src.cols / 2, src.rows / 2)); // 원본 크기의 절반으로 조정

    // 워터마크 삽입할 시작 좌표 (중앙 배치)
    int x_offset = (src.cols - resizedWatermark.cols) / 2;
    int y_offset = (src.rows - resizedWatermark.rows) / 2;

    // 워터마크 삽입 영역 정의
    Mat roi = src(Rect(x_offset, y_offset, resizedWatermark.cols, resizedWatermark.rows));

    if (resizedWatermark.channels() == 4) {
        vector<Mat> channels(4);
        split(resizedWatermark, channels);

        vector<Mat> bgrChannels = { channels[0], channels[1], channels[2] };
        merge(bgrChannels, resizedWatermark);
    }

    // 투명도 합성 (addWeighted)
    double beta = 1.0 - alpha; // 원본 이미지의 가중치
    addWeighted(resizedWatermark, alpha, roi, beta, 0.0, roi); // 워터마크 합성
}

Mat attack4(Mat& src) { // JPEG 압축

    if (compressRate == 100)
        return src;

    vector<int> compressionParams;
    compressionParams.push_back(IMWRITE_JPEG_QUALITY);
    compressionParams.push_back(compressRate); // JPEG 품질 (0~100, 낮을수록 압축률 높음)
    imwrite("watermarked.jpg", src, compressionParams);

    // 압축된 이미지 로드
    Mat compressedImage = imread("watermarked.jpg");
    return compressedImage;
}
