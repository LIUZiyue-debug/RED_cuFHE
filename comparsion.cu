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
uint32_t kNumTests;
PubKey bk;

vector<vector<pair<int, int>>> requests;

int bitCount = 0; 
int bitCalCount_Not = 0; 
int StreamCount = 0;

static Ctxt bit_0;
static Ctxt bit_1;




void adder_32(Ctxt *sum, Ctxt *a, Ctxt *b, Stream *st) {
    static Ctxt G[6][32];  
    static Ctxt P[6][32];
    constexpr int inner_threads = 64;

    #pragma omp parallel
    {
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 0; i < 32; i++) {
            And(G[0][i], a[i], b[i], st[i]);             // G[0][i] = a[i] & b[i]
            //Xor(P[0][i], a[i], b[i], st[i + 32]);        // P[0][i] = a[i] ^ b[i]
            if (i == 12 || i == 24) Synchronize();
        }
        Synchronize();
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 0; i < 32; i++) {
            //And(G[0][i], a[i], b[i], st[i]);             // G[0][i] = a[i] & b[i]
            Xor(P[0][i], a[i], b[i], st[i + 32]);        // P[0][i] = a[i] ^ b[i]
            if (i == 12 || i == 24) Synchronize();
        }
        Synchronize();

        for (int k = 1; k < 6; k++) {
            int distance = 1 << (k - 1); 
            static Ctxt temp[32];        

            #pragma omp parallel for schedule(static) num_threads(inner_threads)
            for (int i = 0; i < 32; i++) {
                if (i >= distance) {
                    And(temp[i], P[k - 1][i], G[k - 1][i - distance], st[i]); 
                    Or(G[k][i], G[k - 1][i], temp[i], st[i]);                  

                    And(P[k][i], P[k - 1][i], P[k - 1][i - distance], st[i + 32]); 
                } else {
                    Copy(G[k][i], G[k - 1][i], st[i]);       
                    Copy(P[k][i], P[k - 1][i], st[i + 32]); 
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
        for (int i = 1; i < 32; i++) {
            Xor(sum[i], P[0][i], G[5][i - 1], st[i]); 
            if (i == 15) Synchronize();
        }
        Synchronize();
    }
}


void multiply_32(Ctxt *product, const Ctxt *a, const Ctxt *b, Stream *st) {

    static Ctxt shifted_temp[32][32];

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < 32 * 32; idx++) {
        And(shifted_temp[idx / 32][idx % 32], a[idx % 32], b[idx / 32], st[idx]);
    }
    Synchronize();

    
    #pragma omp parallel for schedule(static)
    for (int idx = 32; idx < 32 * 32; idx++) {
        if (idx % 32 < idx / 32) {
            Copy(shifted_temp[idx / 32][idx % 32], bit_0, st[idx % 32]);
        } else {
            Copy(shifted_temp[idx / 32][idx % 32], shifted_temp[idx / 32][idx % 32 - idx / 32], st[idx % 32]);
          } 
     }
    Synchronize();


    omp_set_nested(1);
    omp_set_max_active_levels(2);
    constexpr int outer_threads = 4; 

    for (int layer = 1; layer < 6; layer++) {
        int step = (1 << layer);
        
        #pragma omp parallel for schedule(static) num_threads(outer_threads)
        for (int i = 0; i < 32; i += step) {
            adder_32(shifted_temp[i], 
                     shifted_temp[i + step / 2], 
                     shifted_temp[i], 
                     st + 160 * i);
        }
    }
    Synchronize();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < 32; i++) {
        Copy(product[i], shifted_temp[0][i], st[i]);
    }
    Synchronize();
}

void subtractor_32(Ctxt *diff, Ctxt *a, Ctxt *b, Stream *st) {
    static Ctxt G[6][32];  
    static Ctxt P[6][32];
    static Ctxt b_neg[32];
    constexpr int inner_threads = 64;

    #pragma omp parallel
    {
        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 0; i < 32; i++) {
            Not(b_neg[i], b[i], st[i]);
        }
        Synchronize();

        #pragma omp parallel for schedule(static) num_threads(inner_threads)
        for (int i = 0; i < 32; i++) {
            And(G[0][i], a[i], b_neg[i], st[i]);
            Xor(P[0][i], a[i], b_neg[i], st[i + 32]);
            if (i == 12 || i == 24) Synchronize();
        }
        Synchronize();

        for (int k = 1; k < 6; k++) {
            int distance = 1 << (k - 1);
            static Ctxt temp[32];

            #pragma omp parallel for schedule(static) num_threads(inner_threads)
            for (int i = 0; i < 32; i++) {
                if (i >= distance) {
                    And(temp[i], P[k - 1][i], G[k - 1][i - distance], st[i]); 
                    Or(G[k][i], G[k - 1][i], temp[i], st[i]);                  
                    And(P[k][i], P[k - 1][i], P[k - 1][i - distance], st[i + 32]);
                } else {
                    Copy(G[k][i], G[k - 1][i], st[i]);       
                    Copy(P[k][i], P[k - 1][i], st[i + 32]); 
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
        for (int i = 1; i < 32; i++) {
            Xor(diff[i], P[0][i], G[5][i - 1], st[i]); 
            if (i == 15) Synchronize();
        }
        Synchronize();
    }
}

void vectorSubtractor(Ctxt result[3][32], Ctxt a[3][32], Ctxt b[3][32], Stream *st) {
    for (int i = 0; i < 3; ++i) {
        subtractor_32(result[i], a[i], b[i], st);
    }
}

void vectorCross(Ctxt result[3][32], Ctxt a[3][32], Ctxt b[3][32], Stream *st) {
    Ctxt temp1[32], temp2[32];
    multiply_32(temp1, a[1], b[2], st);
    multiply_32(temp2, a[2], b[1], st);
    subtractor_32(result[0], temp1, temp2, st);
    multiply_32(temp1, a[2], b[0], st);
    multiply_32(temp2, a[0], b[2], st);
    subtractor_32(result[1], temp1, temp2, st);
    multiply_32(temp1, a[0], b[1], st);
    multiply_32(temp2, a[1], b[0], st);
    subtractor_32(result[2], temp1, temp2, st);
}

void vectorDot(Ctxt result[32], Ctxt a[3][32], Ctxt b[3][32], Stream *st) {
    Ctxt partial[32];
    for (int i = 0; i < 32; i++) Copy(result[i], bit_0);
    
    for (int i = 0; i < 3; ++i) {
        multiply_32(partial, a[i], b[i], st);
        adder_32(result, result, partial, st);
    }
}

void computeUVAndDet(Ctxt O[3][32], Ctxt D[3][32], Ctxt V0[3][32], Ctxt V1[3][32], Ctxt V2[3][32], Ctxt u_plus_v[32], Ctxt det[32], Stream *st) {
    Ctxt E1[3][32], E2[3][32];
    Ctxt P[3][32], T[3][32], Q[3][32];
    Ctxt u[32], v[32];
    vectorSubtractor(E1, V1, V0, st);
    vectorSubtractor(E2, V2, V0, st);
    vectorCross(P, D, E2, st);
    vectorDot(det, E1, P, st);
    vectorSubtractor(T, O, V0, st);
    vectorDot(u, T, P, st);
    vectorCross(Q, T, E1, st);
    vectorDot(v, D, Q, st);
    adder_32(u_plus_v, u, v, st);
}



void Add32BitEncrypted(Ctxt *sum, Ctxt *A, Stream *st) {
    int n = 32;
    Ctxt* tempSum = new Ctxt[n];
    Ctxt* tempCarry = new Ctxt[n];
    Ctxt* newCarry = new Ctxt[n];

    
    for (int i = 0; i < n; ++i) {
        Xor(tempSum[i], A[i], A[i + 32], st[i]);
        And(tempCarry[i], A[i], A[i + 32], st[i]);
        
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

    for (int k = 0; k < 32; k++) {
            Or(final_answer_Para[k], answers[0][k], answers[0][k + kNumTests], st[0][k]);
            Not(final_answer_Para[k], final_answer_Para[k], st[0][k]);
            //cout << "cuda stream is : " << st[0][k].st() << endl;
          }
}

void RedbitCompareGreatPara(int kNumTests,Ctxt *final_answer_Para, Ctxt **answers, Stream **st){
	if (!final_answer_Para || !answers || !st) {
        cerr << "Invalid input pointers!" << endl;
        return;
    }
	for (int k = 0; k < 32; k++) {
		Not(answers[0][k + kNumTests], answers[0][k + kNumTests], st[0][k + kNumTests]);
		And(final_answer_Para[k], answers[0][k], answers[0][k + kNumTests], st[0][k + kNumTests]);
	}
}

void RedbitCompareLessPara(int kNumTests,Ctxt *final_answer_Para, Ctxt **answers, Stream **st){
	if (!final_answer_Para || !answers || !st) {
        cerr << "Invalid input pointers!" << endl;
        return;
    }
	for (int k = 0; k < 32; k++) {
		Not(answers[0][k], answers[0][k], st[0][k + 2 *  kNumTests]);
		And(final_answer_Para[k], answers[0][k], answers[0][k + kNumTests], st[0][k + 2 * kNumTests]);
	}
}


void server(int shares, uint32_t kNumSMs, int idx, Ctxt** answers, Ctxt* final_answer, Stream** st) {
  while(1) {
    cout << 1 << endl;
    for (int i = 0; i < shares; i++) {
    cout << 2 << endl;
      // check for assignment
      if (requests[idx][i].first != -1) {
       //cout << 2 << endl;
        // terminate upon kill signal (-2)
        if (requests[idx][i].first == -2) {
          //cout << 2 << endl;
          Synchronize();
          return;
        }
        // Perform gate computations
        //RedbitCompareEqual(final_answer,answers[requests[idx][i].second],answers[requests[idx][i].first]);
        // clear assignment
        requests[idx][i].first = -1;
        requests[idx][i].second = -1;
      }
    }
  }
}


void setup(uint32_t kNumSMs, Ctxt** inputs, Ptxt** pt, Stream** st, int idx) {
  cudaSetDevice(idx);
  //cout << "set up idx is : " << idx << endl;
  // send bootstrapping key to GPU
  Initialize(bk);

  // create CUDA streams for the GPU
  st[idx] = new Stream[7520];
  for (int i = 0; i < 7520; i++) {
    st[idx][i].Create();
  }
  Synchronize();

  // Allocate memory for ciphertexts and encrypt
  (*inputs) = new Ctxt[2 * kNumTests];
  for (int i = 0; i < 2 * kNumTests; i++) {
    Encrypt((*inputs)[i], pt[idx][i], pri_key);
  }
  Synchronize();
  return;
}


int main(){

    srand((unsigned)time(NULL));
    
    // get GPU stats (WARNING: assumes all GPUs have the samfinal_answer_Para[k]e number of SMs)
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    //uint32_t kNumSMs = prop.multiProcessorCount;
    //kNumTests = 6272;
    kNumTests = 32;
    
    // get number of available GPUs
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    //cout << "Available GPUs = " << numGPUs << endl;
    
    // create 2D array of plaintext and streams
    //Ptxt* pt[numGPUs];
    Stream* st;
    Ptxt O_p[3][32];
    Ptxt D_p[3][32], V0_p[3][32], V1_p[3][32], V2_p[3][32];
    Ctxt u_plus_v[32];
    Ctxt det[32];
  
    // generate keyset
    SetSeed();
    PriKeyGen(pri_key);
    PubKeyGen(bk, pri_key);
    
    // generate random ptxts (bits) for each GPU
    //pt[i] = new Ptxt[2 * kNumTests];
    for (int j = 0; j <= 3; j++) {
      for (int i = 0; i <= 32; i++) {
          O_p[j][i] = rand() % Ptxt::kPtxtSpace;
          D_p[j][i] = rand() % Ptxt::kPtxtSpace;
          V0_p[j][i] = rand() % Ptxt::kPtxtSpace;
          V1_p[j][i] = rand() % Ptxt::kPtxtSpace;
          V2_p[j][i] = rand() % Ptxt::kPtxtSpace;
      }
     }

    //Ctxt* final_answer_Equal = new Ctxt[kNumTests];
    //Ctxt* final_answer_Great = new Ctxt[kNumTests];
    //Ctxt* final_answer_Less = new Ctxt[kNumTests];
    //Ctxt* final_answer_Add = new Ctxt[kNumTests];
    //Ctxt* final_answer_Add2 = new Ctxt[kNumTests];
    //Ctxt* final_answer_Mul = new Ctxt[kNumTests];
    
   // Initialize shared vector for thread communication
    int num_threads = numGPUs;
    requests.resize(num_threads);
    for (int i = 0; i < num_threads; i++) {
      requests[i].resize(kNumTests);
      for (int j = 0; j < kNumTests; j++) {
        // each element holds indices of data array
        requests[i][j] = make_pair(-1,-1);
          }
     }

    Ctxt** answers = new Ctxt*[2];
    Ctxt O[3][32];
    Ctxt D[3][32], V0[3][32], V1[3][32], V2[3][32];
    omp_set_num_threads(numGPUs);
  
  
    #pragma omp parallel
     {
      cudaSetDevice(0);
      Initialize(bk);
      
      //setup(kNumTests , &answers[0], pt, st, 0);
      st = new Stream[7520];
      for (int i = 0; i < 7520; i++) {
         st[i].Create();
       }
      Synchronize();
      
      for (int j = 0; j < 3 ; j++){
          for (int i = 0; i < 32; i++) {
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
      //cout << answers[0][0].lwe_sample_device_->n() << endl;
     }
    cout << "set up over " << endl;

    // one worker thread for each GPU and a scheduler thread
    omp_set_num_threads(numGPUs);


    high_resolution_clock::time_point t1 = high_resolution_clock::now();
      /////////////////////////////////////////
      //
     // (RED)cuFHE Dynamic Scheduler
     // Enables automatic allocation of FHE
     // workloads to multiple GPUs
     //
     /////////////////////////////////////////
    
    #pragma omp parallel for shared(st, requests)
    for (int i = 1; i < (num_threads+1); i++) {
     if (i != 0) { // workers
     	 //cout << "worker starts" << endl;
       int thread_id = omp_get_thread_num() - 1;
       cudaSetDevice(thread_id);
       //server(kNumTests, kNumSMs, thread_id, &answers[i-1], final_answer, st);
       //RedbitCompareEqual(final_answer,&answers[i-1][0],&answers[i-1][32]);
       //RedbitCompareEqualPara(kNumTests, final_answer_Equal, answers, st);
       //RedbitCompareGreatPara(kNumTests, final_answer_Great, answers, st);
       //RedbitCompareLessPara(kNumTests, final_answer_Less, answers, st);
       
       
       
       computeUVAndDet(O, D, V0, V1, V2, u_plus_v, det, st);
       cout << "Calculation is over " << endl;
       Synchronize();
       
       //Add32BitEncrypted(final_answer_Add, answers[0], st[0]);
       //cout << "adder one is over" << endl;
       //Synchronize();
       //adder_32(answers[0], final_answer_Add2, st[0]); 
       //cout << "adder is over" << endl;
       //Synchronize();
       //multiply_32(answers[0], final_answer_Mul, st[0]);
       //cout << "Multiplier is over" << endl;
	 //Synchronize();
	 //cout << "worker ends" << endl;
      }
     else { // master thread
      int turn = 1; // indicates target worker
      for (int j = 0; j < (kNumTests*numGPUs); j++) {
        if ((j % kNumTests == 0) && (j > 0)) {
          turn++; // assign to next worker
          if (turn > num_threads) { // excludes scheduler
            turn = 1;
          }
        }
        // assign input 1 as index j of GPU array
        requests[turn-1][j % kNumTests].second = j % (kNumTests);
        // assign input 2 as index j+kNumTests
        requests[turn-1][j % kNumTests].first = ((j%kNumTests)+kNumTests) % (2*kNumTests);
      }
      // check to see if all threads are done
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
      // terminate workers
      for (int j = 0; j < num_threads; j++) {
        for (int k = 0; k < kNumTests; k++) {
          requests[j][k].first = -2;
        }
      }
    }
  }

  //cout << "Gate evals: " << kNumTests*numGPUs << endl;
   
   high_resolution_clock::time_point t2 = high_resolution_clock::now();
   duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
   cout << "Time 1: " << time_span.count() << " seconds" << endl;

  //Ptxt recovered_pt = Ptxt();
  //Decrypt(recovered_pt, answers[0][0], pri_key);
  //cout << "result is :" << recovered_pt.message_ << endl;
  
  
  unsigned long Add_val1=0, Add_val2=0, Mul_val=0;
   //#pragma omp parallel shared(final_answer_Equal)
  //{
    Ptxt* recovered_Mulpt = new Ptxt[kNumTests];
    Ptxt* recovered_Addpt = new Ptxt[kNumTests];
    int thread_num = omp_get_thread_num();
    cudaSetDevice(thread_num);
    /*cout << "answer of Euqal is : " << endl;
    for (int i = 0; i < kNumTests; i++) {
      Decrypt(recovered_pt[i], final_answer_Equal[i], pri_key);
      cout <<  recovered_pt[i].get();
     }
    cout << endl << "answer of Great is : " << endl;
    for (int i = 0; i < kNumTests; i++) {
      Decrypt(recovered_pt[i], final_answer_Great[i], pri_key);
      cout <<  recovered_pt[i].get();
     }
    cout << endl << "answer of Less is : " << endl;
    for (int i = 0; i < kNumTests; i++) {
      Decrypt(recovered_pt[i], final_answer_Less[i], pri_key);
      cout <<  recovered_pt[i].get();
     }
    cout << endl << "answer of Add1 is : " << endl;
    for (int i = 0; i < kNumTests; i++) {
      Decrypt(recovered_pt[i], final_answer_Add[i], pri_key);
      //cout <<  recovered_pt[i].get();
      if (recovered_pt[i].get()) Add_val1 |= (1UL << i);
     }
     
    //cout << endl << "answer of Add is : " << endl;
    for (int i = 0; i < kNumTests; i++) {
      Decrypt(recovered_Addpt[i], final_answer_Add2[i], pri_key);
      //cout <<  recovered_pt[32].get();
      if (recovered_Addpt[i].get()) Add_val2 |= (1UL << i);
     }
     for (int i = kNumTests - 1; i >= 0; i--) {
      //cout <<  recovered_Addpt[i].get();
     }
     
     
    //cout << endl << "answer of Mul is : " << endl;
    for (int i = 0; i < kNumTests; i++) {
      Decrypt(recovered_Mulpt[i], final_answer_Mul[i], pri_key);
      if (recovered_Mulpt[i].get()) Mul_val |= (1UL << i);
     }*/
    cout << endl;
    delete [] recovered_Mulpt;
    delete [] recovered_Addpt;
  //}  

    
    
    high_resolution_clock::time_point t3 = high_resolution_clock::now();
    duration<double> time_span2 = duration_cast<duration<double>>(t3 - t2);
    //cout << "Time 2: " << time_span2.count() << " seconds" << endl;
    
 
    /*unsigned long a_val = 0, b_val = 0, Mul_answer = 0;
    for (int  i = 0; i < kNumTests; i++){
          if (pt[0][i].get()) a_val |= (1UL << i);
          if (pt[0][i + 32].get()) b_val |= (1UL << i);
    }
    Mul_answer = ((a_val * b_val) << 32) >> 32;
    
    cout << "True result is :" 
    	   << a_val + b_val << endl
    	   << "Our answer is  :" 
    	   << Add_val2 << endl << endl;
    cout << " True result is :" 
  	   << Mul_answer << endl
    	   << " Our answer is  :"
    	   << Mul_val << endl;*/
 
    /*for (int i = 0; i < 3; i++) {
    delete [] O_p[i];
    delete [] D_p[i];
    delete [] V0_p[i];
    delete [] V1_p[i];
    delete [] V2_p[i];
     }*/
    // free GPU memory
    CleanUp();
        
    return 0;
}
