#include <cmath>
#include <iostream>
#include "ThreeDiagMatrices.h"
#include "vector"
using namespace std;

vector<double> ThomasAlgorithm(vector<double> a,
                               vector<double> b,
                               vector<double> c,
                               vector<double> f,
                               int n){

    vector<double> P(n);
    vector<double> Q(n);
    vector<double> x(n+1);


    P[0] = c[0]/b[0];
    Q[0] = f[0]/b[0];

    for (int i = 1; i < n; i++){
        P[i] = c[i]/(b[i] - a[i]*P[i-1]);
        Q[i] = (f[i] + a[i]*Q[i-1])/(b[i]-a[i]*P[i-1]);
    }

    x[n] = (f[n] + a[n]*Q[n-1])/(b[n]-a[n]*P[n-1]);

    for (int i = n-1; i >= 0; i--){
        x[i] = P[i]*x[i+1] + Q[i];
    }
    return x;
}

vector<double> CycleReductionAlgorithm(vector<double> a,
                                       vector<double> b,
                                       vector<double>c,
                                       vector<double>f,
                                       int n) {
    int q = (int)log2(n);

    vector<double> x(n+1);
    x[0] = f[0];
    x[n] = f[n];

    vector<vector<double>> aBuffer(q, vector<double>(n));
    vector<vector<double>> bBuffer(q, vector<double>(n));
    vector<vector<double>> cBuffer(q, vector<double>(n));
    vector<vector<double>> fBuffer(q, vector<double>(n));

    vector<double> aPrev(n + 1);
    vector<double> bPrev(n + 1);
    vector<double> cPrev(n + 1);
    vector<double> fPrev(n + 1);

    std::copy(a.begin(), a.end(), aBuffer[0].begin());
    std::copy(b.begin(), b.end(), bBuffer[0].begin());
    std::copy(c.begin(), c.end(), cBuffer[0].begin());
    std::copy(f.begin(), f.end(), fBuffer[0].begin());

    std::copy(a.begin(), a.end(), aPrev.begin());
    std::copy(b.begin(), b.end(), bPrev.begin());
    std::copy(c.begin(), c.end(), cPrev.begin());
    std::copy(f.begin(), f.end(), fPrev.begin());

    vector<double> P(n+1);
    vector<double> Q(n+1);

    int size = 1;
    for (int k = 1; k < q; k++) {
        size *= 2;
        int shift = size / 2;
        for (int i = size; i <= n-size; i += size){
            P[i] = aPrev[i] / bPrev[i - shift];
            Q[i] = cPrev[i] / bPrev[i + shift];

            aBuffer[k][i] = P[i] * aPrev[i - shift];
            bBuffer[k][i] = bPrev[i] - P[i] * cPrev[i - shift] - Q[i] * aPrev[i + shift];
            cBuffer[k][i] = Q[i] * cPrev[i + shift];
            fBuffer[k][i] = fPrev[i] + P[i] * fPrev[i - shift] + Q[i] * fPrev[i + shift];
        }

        std::copy(aBuffer[k].begin(), aBuffer[k].end(), aPrev.begin());
        std::copy(bBuffer[k].begin(), bBuffer[k].end(), bPrev.begin());
        std::copy(cBuffer[k].begin(), cBuffer[k].end(), cPrev.begin());
        std::copy(fBuffer[k].begin(), fBuffer[k].end(), fPrev.begin());
    }

    for (int k = q; k > 0; k--) {
        for (int i = size; i <= n-size; i += size*2){
            x[i] = (fBuffer[k - 1][i] + aBuffer[k - 1][i] * x[i - size] + cBuffer[k - 1][i] * x[i + size]) / bBuffer[k - 1][i];
        }
        size /= 2;
    }
    return x;
}