#include <vector>
#include <iostream>
#include <cassert>

class Matrix {
private:
    std::vector<std::vector<float>> data;
    size_t rows;
    size_t cols;

public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<float>(cols, 0.0f));
    }

    Matrix(const std::vector<std::vector<float>>& input) : data(input) {
        rows = input.size();
        cols = rows > 0 ? input[0].size() : 0;
    }

    std::vector<float>& operator[](size_t idx) {
        return data[idx];
    }

    const std::vector<float>& operator[](size_t idx) const {
        return data[idx];
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
};

Matrix conv2d(const Matrix& kernel, const Matrix& input) {
    size_t k_rows = kernel.getRows();
    size_t k_cols = kernel.getCols();
    size_t i_rows = input.getRows();
    size_t i_cols = input.getCols();
    
    // Calculate output dimensions
    size_t out_rows = i_rows - k_rows + 1;
    size_t out_cols = i_cols - k_cols + 1;
    
    if (out_rows <= 0 || out_cols <= 0) {
        throw std::invalid_argument("Invalid kernel size for input");
    }

    Matrix result(out_rows, out_cols);

    for (size_t i = 0; i < out_rows; ++i) {
        for (size_t j = 0; j < out_cols; ++j) {
            float sum = 0.0f;
            for (size_t ki = 0; ki < k_rows; ++ki) {
                for (size_t kj = 0; kj < k_cols; ++kj) {
                    sum += kernel[ki][kj] * input[i + ki][j + kj];
                }
            }
            result[i][j] = sum;
        }
    }

    return result;
}

void test_conv2d() {
    // Test case 1: Simple 3x3 kernel on 5x5 input
    Matrix kernel({
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    });

    Matrix input({
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    });

    Matrix result = conv2d(kernel, input);

    // Check dimensions
    assert(result.getRows() == 3 && result.getCols() == 3);
    
    // Check results (all elements should be -8.0f)
    const float expected = -8.0f;
    const float epsilon = 1e-6f;
    for (size_t i = 0; i < result.getRows(); ++i) {
        for (size_t j = 0; j < result.getCols(); ++j) {
            assert(std::abs(result[i][j] - expected) < epsilon);
        }
    }

    // Test case 2: 4x3 kernel on 6x7 input
    Matrix kernel2({
        {0.3934, 0.3452, 0.4189},
        {0.8490, -0.6760, 0.2487}, 
        {0.2480, 0.1995, -0.0995},
        {0.7166, 0.7923, -0.0937}
    });

    Matrix input2({
        {0.9389, 0.7207, 0.3768, 0.6758, 0.8726, 0.9272, 0.1773},
        {0.4309, 0.0517, 0.8489, 0.0052, 0.3699, 0.7653, 0.1172},
        {0.3916, 0.4775, 0.5562, 0.1186, 0.0821, 0.6072, 0.4001},
        {0.9351, 0.4135, 0.9948, 0.3924, 0.6564, 0.3161, 0.6954},
        {0.2168, 0.9571, 0.6144, 0.3698, 0.5948, 0.5153, 0.8033},
        {0.2271, 0.1548, 0.2779, 0.9592, 0.6943, 0.3582, 0.9837}
    });

    Matrix result2 = conv2d(kernel2, input2);

    // Check dimensions
    assert(result2.getRows() == 3 && result2.getCols() == 5);

    // Expected results
    float expected2[3][5] = {
        {2.359529, 1.4333249, 2.6719432, 1.6571943, 1.3208219},
        {1.7624879, 1.774423, 1.840383, 1.5313119, 1.1334293},
        {1.756585, 0.7700681, 2.097736, 1.6603835, 1.7856003}
    };

    // Check results with epsilon tolerance
    for (size_t i = 0; i < result2.getRows(); ++i) {
        for (size_t j = 0; j < result2.getCols(); ++j) {
            assert(std::abs(result2[i][j] - expected2[i][j]) < epsilon);
        }
    }

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    test_conv2d();
    return 0;
}