#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <cmath>
#include <bitset>
#include <vector>

using namespace cv;
using namespace std;

// �Լ� ����
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

// ������ ���� (strength, ������, numCopies)
// lena512.jpg : (50, 70, 3) // (50, 50, 3)�� ����
// test1.png : (30, 50, 3), (40, 50, 3), (50, 50, 3), (3, 30, 40)  // (30, 40, 3)
// test2.jpg : (20, 80, 5) (30, 80, 5), (40, 80, 5), (50, 80, 5), (60, 90, 5) // (10, 80, 5), (50, 70, 5), (60, 90, 3) ����

int numCopies = 5;
int strength = 30; // ���͸�ũ ����

int compressRate = 90; // ���� ����, ������� 100�� ���� �״�� ��ȯ�ϰ� ��.
float noise_std = 50; // noise ����(ǥ������)


// �⺻�����δ� �����̳� ������ �ּ�ó���Ǿ� �־ ��Ȳ�� �°� �ּ��� ��������� ��.

int main() {
    Mat image = imread("IU.jpg"); // Imgae ���� : lena.jpg  landscape.jpg  IU.jpg
    if (image.empty()) {
        cerr << "�̹����� �ҷ��� �� �����ϴ�!" << endl;
        return -1;
    }

    imshow("source image", image);

    // ���ڿ� ���͸�ũ
    string watermarkText = "192.168.0.1 Choijinu";
    vector<int> watermarkBits = stringToBits(watermarkText);

    unsigned int seed = 42; // seed ��

    // ���͸�ũ ����
    Mat watermarkedImage = image.clone();
    embedWatermark(watermarkedImage, watermarkBits, strength, seed);

    imshow("watermarked image", watermarkedImage);

    printf("PSNR : %f\n", calculatePSNR(image, watermarkedImage)); // ���͸�ũ ������ �̹������� PSNR

    // ���ݵ�
    //attack1(watermarkedImage, noise_std); // ����þ� ������
    //attack2(watermarkedImage); // �׸� �׸���
    //attack3(watermarkedImage, 0.3); // ���ü� ���͸�ũ
    
    Mat compressedImage = attack4(watermarkedImage); // ������� 100�ϰ�� �״�� ��ȯ�ϰ� �Ǿ�����.
    
    //imshow("after attack", watermarkedImage); // ���ݰ�� Ȯ�ο� �ڵ�



    // ���͸�ũ ����
    vector<int> extractedBits = extractWatermark(compressedImage, watermarkBits.size(), strength, seed);

    string extractedText = bitsToString(extractedBits);

    // ��� ���
    cout << "���� ���͸�ũ �ؽ�Ʈ: " << watermarkText << endl;
    cout << "����� ���͸�ũ �ؽ�Ʈ: " << extractedText << endl;

    imshow("final Image", compressedImage);

    printf("PSNR2 : %f\n", calculatePSNR(image, compressedImage)); // ���ݱ��� ������ �̹������� PSNR

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
    RNG rng(seed); // ���� ����
    Mat imageYCrCb;
    cvtColor(src, imageYCrCb, COLOR_BGR2YCrCb);

    vector<Mat> channels;
    split(imageYCrCb, channels);

    // Y ä�θ� ���
    Mat& Y = channels[0];

    for (int i = 0; i < watermark.size(); ++i) {
        for (int j = 0; j < numCopies; ++j) {
            int x = rng.uniform(0, Y.rows / 8) * 8; // 8x8 ��� ����
            int y = rng.uniform(0, Y.cols / 8) * 8;

            Mat block = Y(Rect(y, x, 8, 8));
            Mat dctBlock;
            applyDCT(block, dctBlock);

            // ���͸�ũ ���� (DCT ��� �߰� ��ġ ����)
            dctBlock.at<float>(2, 2) += strength * (watermark[i] == 1 ? 1 : -1);

            // �� DCT ����
            applyIDCT(dctBlock, block);
        }
    }

    merge(channels, imageYCrCb);
    cvtColor(imageYCrCb, src, COLOR_YCrCb2BGR);
}

vector<int> extractWatermark(Mat& src, int watermarkSize, int strength, unsigned int seed) {
    RNG rng(seed); // ���� ����
    Mat imageYCrCb;
    cvtColor(src, imageYCrCb, COLOR_BGR2YCrCb);

    vector<Mat> channels;
    split(imageYCrCb, channels);

    // Y ä�θ� ���
    Mat& Y = channels[0];

    vector<int> extractedWatermark(watermarkSize);

    for (int i = 0; i < watermarkSize; ++i) {
        int votes = 0; // �ټ����� ���� ��ǥ ����
        for (int j = 0; j < numCopies; ++j) {
            int x = rng.uniform(0, Y.rows / 8) * 8; // 8x8 ��� ����
            int y = rng.uniform(0, Y.cols / 8) * 8;

            Mat block = Y(Rect(y, x, 8, 8));
            Mat dctBlock;
            applyDCT(block, dctBlock);

            // ���͸�ũ ��Ʈ ���� �� ��ǥ
            votes += (dctBlock.at<float>(2, 2) > 0) ? 1 : -1;
        }

        // �ټ��ῡ ���� ���͸�ũ ��Ʈ ����
        extractedWatermark[i] = (votes > 0) ? 1 : 0;
    }

    return extractedWatermark;
}


// PSNR ���
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
        bitset<8> charBits(c); // 1���ڸ� 8��Ʈ�� ��ȯ
        for (int i = 7; i >= 0; --i) {
            bits.push_back(charBits[i]);
        }
    }
    return bits;
}

// ��Ʈ�� -> ���ڿ� ��ȯ
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

// ����þ� ������
void attack1(Mat& src, float stddev) {
    cv::Mat noise = cv::Mat(src.size(), src.type());

    // ����þ� ������ ���� (��հ�: mean, ǥ������: stddev)
    cv::randn(noise, 0, stddev);

    src += noise;
}

// �簢�� �׸� �׸���
void attack2(Mat& src) {
    cv::rectangle(src, Point(200, 200), Point(300, 300), Scalar(0, 0, 0), 5, cv::LINE_8);
}

void attack3(Mat& src, double alpha) { // alpha: ���͸�ũ ���� (0: ������ ����, 1: ������)
    Mat visibleWatermark = imread("symbol1.png", IMREAD_UNCHANGED);
    if (visibleWatermark.empty()) {
        cerr << "���͸�ũ �̹����� �ҷ��� �� �����ϴ�!" << endl;
        return;
    }

    imshow("visible watermark", visibleWatermark);

    // ���͸�ũ ũ�� ���� (���� �̹����� �°�)
    Mat resizedWatermark;
    resize(visibleWatermark, resizedWatermark, Size(src.cols / 2, src.rows / 2)); // ���� ũ���� �������� ����

    // ���͸�ũ ������ ���� ��ǥ (�߾� ��ġ)
    int x_offset = (src.cols - resizedWatermark.cols) / 2;
    int y_offset = (src.rows - resizedWatermark.rows) / 2;

    // ���͸�ũ ���� ���� ����
    Mat roi = src(Rect(x_offset, y_offset, resizedWatermark.cols, resizedWatermark.rows));

    if (resizedWatermark.channels() == 4) {
        vector<Mat> channels(4);
        split(resizedWatermark, channels);

        vector<Mat> bgrChannels = { channels[0], channels[1], channels[2] };
        merge(bgrChannels, resizedWatermark);
    }

    // ���� �ռ� (addWeighted)
    double beta = 1.0 - alpha; // ���� �̹����� ����ġ
    addWeighted(resizedWatermark, alpha, roi, beta, 0.0, roi); // ���͸�ũ �ռ�
}

Mat attack4(Mat& src) { // JPEG ����

    if (compressRate == 100)
        return src;

    vector<int> compressionParams;
    compressionParams.push_back(IMWRITE_JPEG_QUALITY);
    compressionParams.push_back(compressRate); // JPEG ǰ�� (0~100, �������� ����� ����)
    imwrite("watermarked.jpg", src, compressionParams);

    // ����� �̹��� �ε�
    Mat compressedImage = imread("watermarked.jpg");
    return compressedImage;
}
