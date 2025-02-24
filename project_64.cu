#include "include/redcufhe_gpu.cuh"
#include "include/redcufhe.h"
#include "include/redcufhe_bootstrap_gpu.cuh"
#include "include/ntt_gpu/ntt.cuh"
#include "include/details/error_gpu.cuh"

using namespace redcufhe;
#include <omp.h>
#include <stdlib.h>
#include <cstdlib>
#include <time.h>
#include <utility>
#include <vector>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <fstream>
#include <sstream>

using namespace std;
using namespace std::chrono;
PriKey pri_key;

uint64_t kNumTests;
PubKey bk;

vector<vector<pair<int, int>>> requests;

int bitCount = 0;
int bitCalCount_Not = 0;
int StreamCount = 0;
static Ctxt bit_0;
static Ctxt bit_1;

void adder_64(Ctxt *sum, Ctxt *a, Ctxt *b, Stream *st) {
    static Ctxt G[6][64];
    static Ctxt P[6][64];
    constexpr int inner_threads = 64 * 2;
    #pragma omp parallel
    {
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 0; i < 64; i++) {
            And(G[0][i], a[i], b[i], st[i]);
            if (i == 12 || i == 24) Synchronize();
        }
        Synchronize();
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 0; i < 64; i++) {
            Xor(P[0][i], a[i], b[i], st[i + 64]);
            if (i == 12 || i == 24) Synchronize();
        }
        Synchronize();
        for (int k = 1; k < 6; k++) {
            int distance = 1 << (k - 1);
            static Ctxt temp[64];
            #pragma omp parallel for schedule(static) num_threads(inner_threads)
            for (int i = 0; i < 64; i++) {
                if (i >= distance) {
                    And(temp[i], P[k - 1][i], G[k - 1][i - distance], st[i]);
                    Or(G[k][i], G[k - 1][i], temp[i], st[i]);
                    And(P[k][i], P[k - 1][i], P[k - 1][i - distance], st[i + 64]);
                } else {
                    Copy(G[k][i], G[k - 1][i], st[i]);
                    Copy(P[k][i], P[k - 1][i], st[i + 64]);
                }
                if (i == 15) Synchronize();
            }
            Synchronize();
        }
        #pragma omp single
        {
            Copy(sum[0], P[0][0], st[0]);
        }
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 1; i < 64; i++) {
            Xor(sum[i], P[0][i], G[5][i - 1], st[i]);
            if (i == 15) Synchronize();
        }
        Synchronize();
    }
}

void multiply_64(Ctxt *product, const Ctxt *a, const Ctxt *b, Stream *st) {
    static Ctxt shifted_temp[64][64];
    constexpr int inner_threads = 64;
    #pragma omp parallel for schedule(static) num_threads(inner_threads)
    for (int idx = 0; idx < 64 * 64; idx++) {
        And(shifted_temp[idx / 64][idx % 64], a[idx % 64], b[idx / 64], st[idx]);
    }
    Synchronize();
    #pragma omp parallel for schedule(static) num_threads(inner_threads)
    for (int idx = 64; idx < 64 * 64; idx++) {
        if (idx % 64 < idx / 64) {
            Copy(shifted_temp[idx / 64][idx % 64], bit_0, st[idx % 64]);
        } else {
            Copy(shifted_temp[idx / 64][idx % 64], shifted_temp[idx / 64][idx % 64 - idx / 64], st[idx % 64]);
        }
    }
    Synchronize();
    omp_set_nested(1);
    omp_set_max_active_levels(2);
    constexpr int outer_threads = 4;
    for (int layer = 1; layer < 6; layer++) {
        int step = (1 << layer);
        #pragma omp parallel for schedule(static) num_threads(outer_threads)
        for (int i = 0; i < 64; i += step) {
            adder_64(shifted_temp[i],
                     shifted_temp[i + step / 2],
                     shifted_temp[i],
                     st + 64 * 2 * i);
        }
    }
    Synchronize();
    #pragma omp parallel for schedule(static) num_threads(inner_threads)
    for (int i = 0; i < 64; i++) {
        Copy(product[i], shifted_temp[0][i], st[i]);
    }
    Synchronize();
}

void subtractor_64(Ctxt *diff, Ctxt *a, Ctxt *b, Stream *st) {
    static Ctxt G[6][64];
    static Ctxt P[6][64];
    static Ctxt b_neg[64];
    constexpr int inner_threads = 64 * 2;
    #pragma omp parallel
    {
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 0; i < 64; i++) {
            Not(b_neg[i], b[i], st[i]);
        }
        Synchronize();
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 0; i < 64; i++) {
            And(G[0][i], a[i], b_neg[i], st[i]);
            Xor(P[0][i], a[i], b_neg[i], st[i + 64]);
            if (i == 12 || i == 24) Synchronize();
        }
        Synchronize();
        for (int k = 1; k < 6; k++) {
            int distance = 1 << (k - 1);
            static Ctxt temp[64];
            #pragma omp parallel for schedule(static) num_threads(inner_threads)
            for (int i = 0; i < 64; i++) {
                if (i >= distance) {
                    And(temp[i], P[k - 1][i], G[k - 1][i - distance], st[i]);
                    Or(G[k][i], G[k - 1][i], temp[i], st[i]);
                    And(P[k][i], P[k - 1][i], P[k - 1][i - distance], st[i + 64]);
                } else {
                    Copy(G[k][i], G[k - 1][i], st[i]);
                    Copy(P[k][i], P[k - 1][i], st[i + 64]);
                }
                if (i == 15) Synchronize();
            }
            Synchronize();
        }
        #pragma omp single
        {
            Copy(diff[0], P[0][0], st[0]);
            Or(diff[0], diff[0], bit_1, st[0]);
        }
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 1; i < 64; i++) {
            Xor(diff[i], P[0][i], G[5][i - 1], st[i]);
            if (i == 15) Synchronize();
        }
        Synchronize();
    }
}

void vectorSubtractor(Ctxt result[3][64], Ctxt a[3][64], Ctxt b[3][64], Stream *st) {
    for (int i = 0; i < 3; ++i) {
        subtractor_64(result[i], a[i], b[i], st);
    }
}

void vectorCross(Ctxt result[3][64], Ctxt a[3][64], Ctxt b[3][64], Stream *st) {
    Ctxt temp1[64], temp2[64];
    multiply_64(temp1, a[1], b[2], st);
    multiply_64(temp2, a[2], b[1], st);
    subtractor_64(result[0], temp1, temp2, st);
    multiply_64(temp1, a[2], b[0], st);
    multiply_64(temp2, a[0], b[2], st);
    subtractor_64(result[1], temp1, temp2, st);
    multiply_64(temp1, a[0], b[1], st);
    multiply_64(temp2, a[1], b[0], st);
    subtractor_64(result[2], temp1, temp2, st);
}

void vectorDot(Ctxt result[64], Ctxt a[3][64], Ctxt b[3][64], Stream *st) {
    Ctxt partial[64];
    for (int i = 0; i < 64; i++) Copy(result[i], bit_0);
    for (int i = 0; i < 3; ++i) {
        multiply_64(partial, a[i], b[i], st);
        adder_64(result, result, partial, st);
    }
}

void computeUVAndDet(Ctxt O[3][64], Ctxt D[3][64], Ctxt V0[3][64], Ctxt V1[3][64], Ctxt V2[3][64], Ctxt u_plus_v[64], Ctxt det[64], Stream *st) {
    Ctxt E1[3][64], E2[3][64];
    Ctxt P[3][64], T[3][64], Q[3][64];
    Ctxt u[64], v[64];
    vectorSubtractor(E1, V1, V0, st);
    vectorSubtractor(E2, V2, V0, st);
    vectorCross(P, D, E2, st);
    vectorDot(det, E1, P, st);
    vectorSubtractor(T, O, V0, st);
    vectorDot(u, T, P, st);
    vectorCross(Q, T, E1, st);
    vectorDot(v, D, Q, st);
    adder_64(u_plus_v, u, v, st);
}

void Add64BitEncrypted(Ctxt *sum, Ctxt *A, Stream *st) {
    int n = 64;
    Ctxt* tempSum = new Ctxt[n];
    Ctxt* tempCarry = new Ctxt[n];
    Ctxt* newCarry = new Ctxt[n];
    for (int i = 0; i < n; ++i) {
        Xor(tempSum[i], A[i], A[i + 64], st[i]);
        And(tempCarry[i], A[i], A[i + 64], st[i]);
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
    for (int i = 0; i < n; ++i) {
        Copy(sum[i], tempSum[i], st[i % n]);
    }
}

void RedbitCompareEqualPara (int kNumTests,Ctxt *final_answer_Para, Ctxt **answers, Stream **st){
    if (!final_answer_Para || !answers || !st) {
        cerr << "Invalid input pointers!" << endl;
        return;
    }
    for (int k = 0; k < 64; k++) {
        Or(final_answer_Para[k], answers[0][k], answers[0][k + kNumTests], st[0][k]);
        Not(final_answer_Para[k], final_answer_Para[k], st[0][k]);
    }
}

void RedbitCompareGreatPara(int kNumTests,Ctxt *final_answer_Para, Ctxt **answers, Stream **st){
    if (!final_answer_Para || !answers || !st) {
        cerr << "Invalid input pointers!" << endl;
        return;
    }
    for (int k = 0; k < 64; k++) {
        Not(answers[0][k + kNumTests], answers[0][k + kNumTests], st[0][k + kNumTests]);
        And(final_answer_Para[k], answers[0][k], answers[0][k + kNumTests], st[0][k + kNumTests]);
    }
}

void RedbitCompareLessPara(int kNumTests,Ctxt *final_answer_Para, Ctxt **answers, Stream **st){
    if (!final_answer_Para || !answers || !st) {
        cerr << "Invalid input pointers!" << endl;
        return;
    }
    for (int k = 0; k < 64; k++) {
        Not(answers[0][k], answers[0][k], st[0][k + 2 *  kNumTests]);
        And(final_answer_Para[k], answers[0][k], answers[0][k + kNumTests], st[0][k + 2 * kNumTests]);
    }
}

void server(int shares, uint64_t kNumSMs, int idx, Ctxt** answers, Ctxt* final_answer, Stream** st) {
    while(1) {
        cout << 1 << endl;
        for (int i = 0; i < shares; i++) {
            cout << 2 << endl;
            if (requests[idx][i].first != -1) {
                if (requests[idx][i].first == -2) {
                    Synchronize();
                    return;
                }
                requests[idx][i].first = -1;
                requests[idx][i].second = -1;
            }
        }
    }
}

void setup(uint64_t kNumSMs, Ctxt** inputs, Ptxt** pt, Stream** st, int idx) {
    cudaSetDevice(idx);
    Initialize(bk);
    st[idx] = new Stream[2048];
    for (int i = 0; i < 2048; i++) {
        st[idx][i].Create();
    }
    Synchronize();
    (*inputs) = new Ctxt[2 * kNumTests];
    for (int i = 0; i < 2 * kNumTests; i++) {
        Encrypt((*inputs)[i], pt[idx][i], pri_key);
    }
    Synchronize();
    return;
}

int main(){
    srand((unsigned)time(NULL));
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    kNumTests = 64;
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    Stream* st;
    Ptxt O_p[3][64];
    Ptxt D_p[3][64], V0_p[3][64], V1_p[3][64], V2_p[3][64];
    Ctxt u_plus_v[64];
    Ctxt det[64];
    SetSeed();
    PriKeyGen(pri_key);
    PubKeyGen(bk, pri_key);
    for (int j = 0; j <= 3; j++) {
        for (int i = 0; i <= 64; i++) {
            O_p[j][i] = rand() % Ptxt::kPtxtSpace;
            D_p[j][i] = rand() % Ptxt::kPtxtSpace;
            V0_p[j][i] = rand() % Ptxt::kPtxtSpace;
            V1_p[j][i] = rand() % Ptxt::kPtxtSpace;
            V2_p[j][i] = rand() % Ptxt::kPtxtSpace;
        }
    }
    int num_threads = numGPUs;
    requests.resize(num_threads);
    for (int i = 0; i < num_threads; i++) {
        requests[i].resize(kNumTests);
        for (int j = 0; j < kNumTests; j++) {
            requests[i][j] = make_pair(-1,-1);
        }
    }
    Ctxt** answers = new Ctxt*[2];
    Ctxt O[3][64];
    Ctxt D[3][64], V0[3][64], V1[3][64], V2[3][64];
    omp_set_num_threads(numGPUs);
    high_resolution_clock::time_point t0 = high_resolution_clock::now();

    #pragma omp parallel
    {
        cudaSetDevice(0);
        Initialize(bk);
        st = new Stream[4096];
        for (int i = 0; i < 4096; i++) {
            st[i].Create();
        }
        Synchronize();
        for (int j = 0; j < 3 ; j++){
            for (int i = 0; i < 64; i++) {
                Encrypt(O[j][i], O_p[j][i], pri_key);
                Encrypt(D[j][i], D_p[j][i], pri_key);
                Encrypt(V0[j][i], V0_p[j][i], pri_key);
                Encrypt(V1[j][i], V1_p[j][i], pri_key);
                Encrypt(V2[j][i], V2_p[j][i], pri_key);
            }
        }
        Synchronize();
        ConstantRed(bit_0, 0);
        ConstantRed(bit_1, 1);
    }
    cout << "set up over " << endl;
    omp_set_num_threads(numGPUs);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    duration<double> time_span0 = duration_cast<duration<double>>(t1 - t0);
    cout << "Time 0: " << time_span0.count() << " seconds" << endl;

    high_resolution_clock::time_point t2, t3;
    #pragma omp parallel for shared(st, requests)
    for (int i = 1; i < (num_threads+1); i++) {
        if (i != 0) {
            int thread_id = omp_get_thread_num() - 1;
            cudaSetDevice(thread_id);
            //computeUVAndDet(O, D, V0, V1, V2, u_plus_v, det, st);
            adder_64(det, O[0], D[0], st);
            t2 = high_resolution_clock::now();

            //multiply_64(u_plus_v, O[0], D[0], st);
            t3 = high_resolution_clock::now();

            cout << "Calculation is over " << endl;
            Synchronize();
        }
        else {
            int turn = 1;
            for (int j = 0; j < (kNumTests*numGPUs); j++) {
                if ((j % kNumTests == 0) && (j > 0)) {
                    turn++;
                    if (turn > num_threads) {
                        turn = 1;
                    }
                }
                requests[turn-1][j % kNumTests].second = j % (kNumTests);
                requests[turn-1][j % kNumTests].first = ((j%kNumTests)+kNumTests) % (2*kNumTests);
            }
            bool end = false;
            while (end == false) {
                end = true;
                for (int j = 0; j < num_threads; j++) {
                    for (int k = 0; k < kNumTests; k++) {
                        if (requests[j][k].first != -1) {
                            end = false;
                            break;
                        }
                    }
                }
            }
            for (int j = 0; j < num_threads; j++) {
                for (int k = 0; k < kNumTests; k++) {
                    requests[j][k].first = -2;
                }
            }
        }
    }
    duration<double> time_span1 = duration_cast<duration<double>>(t2 - t1);
    cout << "Time 1: " << time_span1.count() << " seconds" << endl;
    
    duration<double> time_span2 = duration_cast<duration<double>>(t3 - t2);
    cout << "Time 2: " << time_span2.count() << " seconds" << endl;

    Ptxt* recovered_u_plus_v = new Ptxt[64];
    Ptxt* recovered_det = new Ptxt[64];
    
    for (int i = 0; i < kNumTests; i++) {
        Decrypt(recovered_u_plus_v[i], u_plus_v[i], pri_key);
        Decrypt(recovered_det[i], det[i], pri_key);
    }
    
    high_resolution_clock::time_point t4 = high_resolution_clock::now();
    duration<double> time_span3 = duration_cast<duration<double>>(t4 - t3);
    cout << "Time final: " << time_span3.count() << " seconds" << endl;

    delete [] recovered_u_plus_v;
    delete [] recovered_det;

    CleanUp();
    
    cudaDeviceSynchronize();
    return 0;
}
