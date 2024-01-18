#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <mpi.h>

using namespace std;
#define N 30
#define M 50000

// Cấu trúc đại diện cho một điểm dữ liệu
struct DataPoint {
    vector<double> x;
    int label;
};

// Hàm tính khoảng cách Euclidean giữa hai điểm
double calculateEuclideanDistance(const DataPoint& p1, const DataPoint& p2) {
    double distance = 0;
    for (int i = 0; i < N; i++)
    {
        distance += pow(p1.x[i] - p2.x[i], 2);
    }
    return sqrt(distance);
}



int main(int argc, char* argv[]) {
    // Tạo dữ liệu mẫu
    vector<DataPoint> dataset(M);

    srand(time(0));
    for (int i = 0; i < M / 2; ++i) {
        for (int j = 0; j < N; j++)
        {
            double value = double(2 * j - 60 + (rand() % 100) / 100.0);
            dataset[i].x.push_back(value);
        }
        dataset[i].label = 1;
    }
    for (int i = M / 2; i < M; ++i) {
        for (int j = 0; j < N; j++)
        {
            double value = double(2 * j + 60 + (rand() % 100) / 100.0);
            dataset[i].x.push_back(value);
        }
        dataset[i].label = 2;
    }
    // Tạo điểm truy vấn
    DataPoint queryPoint;
    for (int i = 0; i < N; i++)
    {
        queryPoint.x.push_back(500.0);
    }

    // Thiết lập tham số K
    int k = 31;


    // Thực hiện KNN MPI
    auto start = std::chrono::high_resolution_clock::now();
    int rank, size;
    int result;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dataSize = dataset.size();
    int chunkSize = dataSize / size;
    int remainder = dataSize % size;

    // Tính toán khoảng cách từ điểm truy vấn đến mỗi điểm trong tập dữ liệu
    std::vector<std::pair<double, int>> localDistancesAndLabels(chunkSize + (rank == size - 1 ? remainder : 0));
    for (int i = 0; i < localDistancesAndLabels.size(); ++i) {
        int globalIndex = rank * chunkSize + std::min(rank, remainder) + i;
        double distance = calculateEuclideanDistance(queryPoint, dataset[globalIndex]);
        localDistancesAndLabels[i] = { distance, dataset[globalIndex].label };
    }

    // Tạo mảng chứa tất cả các khoảng cách và nhãn
    std::vector<std::pair<double, int>> allDistancesAndLabels(dataSize);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(localDistancesAndLabels.data(), localDistancesAndLabels.size() * sizeof(std::pair<double, int>), MPI_BYTE,
        allDistancesAndLabels.data(), localDistancesAndLabels.size() * sizeof(std::pair<double, int>), MPI_BYTE,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "N: " << M << endl;
        cout << "K: " << k << endl;
        // Sắp xếp khoảng cách và nhãn trên tiến trình chính
        std::sort(allDistancesAndLabels.begin(), allDistancesAndLabels.end(),
            [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
            });

        int countClass1 = 0, countClass2 = 0;

        for (int i = 0; i < k; ++i) {
            if (allDistancesAndLabels[i].second == 1) {
                countClass1++;
            }
            else {
                countClass2++;
            }
        }
        result = (countClass1 > countClass2) ? 1 : 2;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        cout << "Knn MPI Results: " << result << endl;
        std::cout << "Time: " << diff.count() << " s\n";
        
    }
       
    MPI_Finalize();

    return 0;
}