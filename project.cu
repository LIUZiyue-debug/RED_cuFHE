#include "include/redcufhe_gpu.cuh"
#include "include/details/error_gpu.cuh"

#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <utility>
#include <vector>
#include <math.h>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
using namespace redcufhe;
using namespace std;
using namespace std::chrono;

PriKey pri_key;
PubKey bk;
uint32_t kNumTests;

void Add32BitEncrypted(Ctxt *sum, Ctxt *A,Ctxt *B, Stream *st) {
    int n = 32;
    Ctxt* tempSum = new Ctxt[n];
    Ctxt* tempCarry = new Ctxt[n];
    Ctxt* newCarry = new Ctxt[n];

    
    for (int i = 0; i < n; ++i) {
        Xor(tempSum[i], A[i], B[i], st[i]);
        And(tempCarry[i], A[i], B[i], st[i]);
        
    }
      Synchronize();
    for (int step = 1; step < n; step *= 2) {

        for (int i = step; i < n; ++i) {
            Ctxt shiftedCarry;
            Copy(shiftedCarry, tempCarry[i - step], st[i % n]);
            Xor(tempSum[i], tempSum[i], shiftedCarry, st[i % n]);
            And(newCarry[i], tempSum[i], shiftedCarry, st[i % n]);
            Or(newCarry[i], newCarry[i], tempCarry[i], st[i % n]);
        }
        Synchronize();
        for (int i = 0; i < n; ++i) {
            Copy(tempCarry[i], newCarry[i], st[i % n]);
        }
    }
cout << 3 << endl;
    for (int i = 0; i < n; ++i) {
        Copy(sum[i], tempSum[i], st[i % n]);
    }
}

vector<int> Add32BitPlain(const vector<int>& A, const vector<int>& B) {
    size_t n = 32;
    vector<int> sum(n, 0);
    int carry = 0;

    for (size_t i = 0; i < n; ++i) {
        int temp = A[i] + B[i] + carry;
        sum[i] = temp % 2;
        carry = temp / 2;
    }

    return sum;
}

int main() {
    srand(time(NULL));
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t kNumSMs = prop.multiProcessorCount;
    kNumTests = 32;

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    Ptxt* pt[numGPUs];
    Stream* st[numGPUs];

    SetSeed();
    PriKeyGen(pri_key);
    PubKeyGen(bk, pri_key);

    for (int i = 0; i < numGPUs; ++i) {
        pt[i] = new Ptxt[2 * kNumTests];
        for (uint32_t j = 0; j < 2 * kNumTests; ++j) {
            pt[i][j] = rand() % Ptxt::kPtxtSpace;
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < numGPUs; ++i) {
        st[i] = new Stream[kNumTests];
        for (uint32_t j = 0; j < kNumTests; ++j) {
            st[i][j].Create();
        }
    }

    Ctxt* A = new Ctxt[32];
    Ctxt* B = new Ctxt[32];
    Ctxt* Sum = new Ctxt[32];

    for (int i = 0; i < 32; ++i) {
        Encrypt(A[i], pt[0][i], pri_key);
        Encrypt(B[i], pt[0][i + 32], pri_key);
    }
    Synchronize();
    cout << 1 << endl;

    #pragma omp parallel shared(Sum, A, B, st)
    {
        Add32BitEncrypted(Sum, A, B, st[0]);
    }
    

    vector<int> result(32);
    for (int i = 0; i < 32; ++i) {
        Ptxt plaintextResult;
        Decrypt(plaintextResult, Sum[i], pri_key);
        result[i] = plaintextResult.message_;
    }

    vector<int> plainA(32), plainB(32);
    for (int i = 0; i < 32; ++i) {
        plainA[i] = pt[0][i].message_;
        plainB[i] = pt[0][i + 32].message_;
    }
    vector<int> expected = Add32BitPlain(plainA, plainB);

    cout << "Encrypted Sum: ";
    for (int i = 31; i >= 0; --i) {
        cout << result[i];
    }
    cout << endl;

    cout << "Plaintext Sum: ";
    for (int i = 31; i >= 0; --i) {
        cout << expected[i];
    }
    cout << endl;

    return 0;
}

