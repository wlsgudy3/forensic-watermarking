#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <cmath>
#include <bitset>
#include <vector>

using namespace cv;
using namespace std;

int numCopies = 5;

void embedWatermark(Mat& src, const vector<int>& watermark, int strength, unsigned int seed);
vector<int> extractWatermark(Mat& src, int watermarkSize, int strength, unsigned int seed);
void dwt2D(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH);
void idwt2D(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst);
double calculatePSNR(const Mat& original, const Mat& processed);
vector<int> stringToBits(const string& str);
string bitsToString(const vector<int>& bits);


void addGaussianNoise(const Mat& src, Mat& dst, double mean, double stddev) {
    Mat floatSrc;
    src.convertTo(floatSrc, CV_32F);

    Mat noise = Mat::zeros(src.size(), CV_32FC(src.channels()));
    randn(noise, mean, stddev);
    add(floatSrc, noise, floatSrc);

    floatSrc.convertTo(dst, src.type());
}

void testGaussianNoise(const Mat& watermarkedImage, const vector<int>& watermarkBits, int strength, unsigned int seed, float stddev) {
    Mat noisyImage;
    addGaussianNoise(watermarkedImage, noisyImage, 0, stddev); // 가우시안 노이즈 표준편차 설정

    double psnr = calculatePSNR(watermarkedImage, noisyImage);
    cout << "PSNR (Watermarked vs Noisy): " << psnr << endl;
    imshow("Noise Watermark Test", noisyImage);

    vector<int> extractedBits = extractWatermark(noisyImage, watermarkBits.size(), strength, seed);
    string extractedText = bitsToString(extractedBits);
    cout << "Extracted Watermark (Noisy): " << extractedText << endl;
}

void testCompression(const Mat& watermarkedImage, const vector<int>& watermarkBits, int strength, unsigned int seed, int compressionRate) {
    vector<int> compressionParams;
    compressionParams.push_back(IMWRITE_JPEG_QUALITY);
    compressionParams.push_back(compressionRate); // 압축 품질 (0~100, 낮을수록 압축률 높음)
    imwrite("compressed_test.jpg", watermarkedImage, compressionParams);

    Mat compressedImage = imread("compressed_test.jpg");

    double psnr = calculatePSNR(watermarkedImage, compressedImage);
    cout << "PSNR (Watermarked vs Compressed): " << psnr << endl;
    imshow("Compression Watermark Test", compressedImage);

    vector<int> extractedBits = extractWatermark(compressedImage, watermarkBits.size(), strength, seed);
    string extractedText = bitsToString(extractedBits);
    cout << "Extracted Watermark (Compressed): " << extractedText << endl;
}

void testVisibleWatermark(Mat& src, const vector<int>& watermarkBits, int strength, unsigned int seed, double alpha) {
    Mat visibleWatermark = imread("kwangwoon.png", IMREAD_UNCHANGED);

    Mat resizedWatermark;
    resize(visibleWatermark, resizedWatermark, Size(src.cols / 2, src.rows / 2));

    int x_offset = (src.cols - resizedWatermark.cols) / 2;
    int y_offset = (src.rows - resizedWatermark.rows) / 2;

    Mat roi = src(Rect(x_offset, y_offset, resizedWatermark.cols, resizedWatermark.rows));

    if (resizedWatermark.channels() == 4) {
        vector<Mat> channels(4);
        split(resizedWatermark, channels);

        vector<Mat> bgrChannels = { channels[0], channels[1], channels[2] };
        merge(bgrChannels, resizedWatermark);
    }

    double beta = 1.0 - alpha;
    Mat watermarkedImage = src.clone();
    addWeighted(resizedWatermark, alpha, roi, beta, 0.0, roi);

    double psnr = calculatePSNR(watermarkedImage, src);
    cout << "PSNR (Watermarked vs Visible Watermarked): " << psnr << endl;
    imshow("Visible Watermark Test", src);

    vector<int> extractedBits = extractWatermark(src, watermarkBits.size(), strength, seed);
    string extractedText = bitsToString(extractedBits);

    cout << "Extracted watermark (Visible Watermark): " << extractedText << endl;
}


int main() {
    Mat image = imread("lena.jpg");
    //Mat image = imread("landscape.jpg");
    //Mat image = imread("IU.jpg");

    imshow("Original Image", image);

    // 워터마크 생성
    const int watermarkSize = 64; // 워터마크 크기
    vector<int> watermark(watermarkSize);
    unsigned int seed = 42; // 난수 생성용 키
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, 1);
    for (int& bit : watermark) {
        bit = dist(rng);
    }

    // 문자열 워터마크(IP 주소, 이름)
    string watermarkText = "192.168.0.1 LeeJungHyun";
    vector<int> watermarkBits = stringToBits(watermarkText);

    // 워터마크 삽입
    int strength = 50; // 워터마크 삽입 강도 (10, 30, 50)
    Mat watermarkedImage = image.clone();
    embedWatermark(watermarkedImage, watermarkBits, strength, seed);

    // 워터마크 삽입된 이미지와 원본 이미지 비교
    printf("PSNR (Original vs Watermarked): %f\n", calculatePSNR(image, watermarkedImage));
    imshow("Watermarked Image", watermarkedImage);

    // 노이즈 테스트
    printf("\n--- Gaussian Noise Test ---\n");
    testGaussianNoise(watermarkedImage, watermarkBits, strength, seed, 1);
    //testGaussianNoise(watermarkedImage, watermarkBits, strength, seed, 10);
    //testGaussianNoise(watermarkedImage, watermarkBits, strength, seed, 50);

    // JPEG 압축 테스트
    printf("\n--- Compression Test ---\n");
    testCompression(watermarkedImage, watermarkBits, strength, seed, 70);
    //testCompression(watermarkedImage, watermarkBits, strength, seed, 80);
    //testCompression(watermarkedImage, watermarkBits, strength, seed, 90);

    // 가시성 워터마크 테스트
    printf("\n--- Visible Watermark Test ---\n");
    testVisibleWatermark(watermarkedImage, watermarkBits, strength, seed, 0.3);

    waitKey(0);
    return 0;
}


// DWT 변환 함수
void dwt2D(const Mat& src, Mat& LL, Mat& LH, Mat& HL, Mat& HH) {
    Mat floatSrc = src;
    if (src.type() != CV_32F) {
        src.convertTo(floatSrc, CV_32F);
    }

    int rows = floatSrc.rows / 2;
    int cols = floatSrc.cols / 2;
    LL = Mat::zeros(rows, cols, CV_32F);
    LH = Mat::zeros(rows, cols, CV_32F);
    HL = Mat::zeros(rows, cols, CV_32F);
    HH = Mat::zeros(rows, cols, CV_32F);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float a = floatSrc.at<float>(i * 2, j * 2);
            float b = floatSrc.at<float>(i * 2, j * 2 + 1);
            float c = floatSrc.at<float>(i * 2 + 1, j * 2);
            float d = floatSrc.at<float>(i * 2 + 1, j * 2 + 1);

            LL.at<float>(i, j) = (a + b + c + d) / 4.0f;
            LH.at<float>(i, j) = (a - b + c - d) / 4.0f;
            HL.at<float>(i, j) = (a + b - c - d) / 4.0f;
            HH.at<float>(i, j) = (a - b - c + d) / 4.0f;
        }
    }
}


// IDWT 변환 함수
void idwt2D(const Mat& LL, const Mat& LH, const Mat& HL, const Mat& HH, Mat& dst) {
    int rows = LL.rows * 2;
    int cols = LL.cols * 2;

    dst = Mat::zeros(rows, cols, CV_32F);

    for (int i = 0; i < LL.rows; i++) {
        for (int j = 0; j < LL.cols; j++) {
            float a = LL.at<float>(i, j);
            float b = LH.at<float>(i, j);
            float c = HL.at<float>(i, j);
            float d = HH.at<float>(i, j);

            dst.at<float>(i * 2, j * 2) = (a + b + c + d);
            dst.at<float>(i * 2, j * 2 + 1) = (a - b + c - d);
            dst.at<float>(i * 2 + 1, j * 2) = (a + b - c - d);
            dst.at<float>(i * 2 + 1, j * 2 + 1) = (a - b - c + d);
        }
    }
}

// 워터마크 삽입 함수
void embedWatermark(Mat& src, const vector<int>& watermark, int strength, unsigned int seed) {
    RNG rng(seed);
    Mat imageYCrCb;
    cvtColor(src, imageYCrCb, COLOR_BGR2YCrCb);

    vector<Mat> channels;
    split(imageYCrCb, channels);

    Mat& Y = channels[0];

    Mat floatY;
    Y.convertTo(floatY, CV_32F);

    Mat LL, LH, HL, HH;
    dwt2D(floatY, LL, LH, HL, HH);

    // HH 대역에 워터마크 삽입
    for (int i = 0; i < watermark.size(); ++i) {
        for (int j = 0; j < numCopies; ++j) {
            int x = rng.uniform(0, HH.rows);
            int y = rng.uniform(0, HH.cols);

            HH.at<float>(x, y) += strength * (watermark[i] == 1 ? 1 : -1);
        }
    }

    idwt2D(LL, LH, HL, HH, floatY);

    floatY.convertTo(Y, CV_8U);

    merge(channels, imageYCrCb);
    cvtColor(imageYCrCb, src, COLOR_YCrCb2BGR);
}

// 워터마크 추출 함수
vector<int> extractWatermark(Mat& src, int watermarkSize, int strength, unsigned int seed) {
    RNG rng(seed);
    Mat imageYCrCb;
    cvtColor(src, imageYCrCb, COLOR_BGR2YCrCb);

    vector<Mat> channels;
    split(imageYCrCb, channels);

    Mat& Y = channels[0];
    Mat LL, LH, HL, HH;
    dwt2D(Y, LL, LH, HL, HH);

    vector<int> extractedWatermark(watermarkSize);

    // 워터마크 추출
    for (int i = 0; i < watermarkSize; ++i) {
        int votes = 0;
        for (int j = 0; j < numCopies; ++j) {
            int x = rng.uniform(0, HH.rows);
            int y = rng.uniform(0, HH.cols);

            votes += (HH.at<float>(x, y) > 0) ? 1 : -1;
        }
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

// 문자열 -> 비트열
vector<int> stringToBits(const string& str) {
    vector<int> bits;
    for (char c : str) {
        bitset<8> charBits(c);
        for (int i = 7; i >= 0; --i) {
            bits.push_back(charBits[i]);
        }
    }
    return bits;
}

// 비트열 -> 문자열
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