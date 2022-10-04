#include <stdio.h>
#include <math.h>

double dotprod (
    int n,
    double *vec1,
    double *vec2
) {
    int k, m;
    double sum;

    sum = 0.0;
    k = n / 4;
    m = n % 4;

    while (k--) {
        sum += *vec1 * *vec2;
        sum += *(vec1+1) * *(vec2+1);
        sum += *(vec1+2) * *(vec2+2);
        sum += *(vec1+3) * *(vec2+3);
        vec1 += 4;
        vec2 += 4;
    }

    while (m--)
        sum += *vec1++ * *vec2++;

    return sum;
}

void activity (
    double *input,  // ninputs 만큼의 길이를 갖는 현재 뉴런의 입력 벡터
    double *coefs,  // ninputs + 1 길이를 갖는 가중치 벡터 (바이러스를 마지막에 추가)
    double *output, // 도달한 현재 뉴런의 활성화 값
    int ninputs,    // 입력 벡터의 길이
    int outlin      // 선형 여부 판단
) {
    double sum;

    sum = dotprod(ninputs, input, coefs);
    sum += coefs[ninputs];  // 바이어스 항

    if (outlin)             // 선형
        *output = sum;
    else                    // 로지스틱
        *output = 1.0 / (1.0 + exp(-sum));
}

static void trial_thr (
    double *input,          // n_model_inputs 만큼의 길이를 갖는 입력 벡터
    int n_all,              // 출력은 포함하고 입력은 제외한 레이어의 개수
    int n_model_inputs,     // 모델에 입력되는 입력의 개수
    double *outputs,        // ntarg 만큼의 길이를 갖는 모델의 출력 벡터
    int ntarg,              // 모델의 최종 출력 개수
    int *nhid_all,          // nhid_all[i]는 i번째 은닉 레이어에 존재하는 은닉 뉴런의 개수
    double *weights_opt[],  // weights_opt[i]는 i번째 은닉 레이어의 가중치 벡터를 가리키는 포인터
    double *hid_act[],      // hid_act[i]는 i번째 은닉 레이어의 활성화 벡터를 가리키는 포인터
    double *final_layer_weights,    // 마지막 레이어의 가중치를 가리키는 포인터
    int classifier          // 0이 아니면 softmax로 출력하며, 1이면 선형 조합으로 출력
) {
    int i, ilayer;
    double sum;

    for (ilayer = 0; ilayer < n_all; ilayer++) {
        if (ilayer == 0 && n_all == 1) {    // 은닉 레이어가 없는 경우
            for (i = 0; i < ntarg; i++)
                activity(
                    input,
                    final_layer_weights + i * (n_model_inputs + 1),
                    outputs + i,
                    n_model_inputs,
                    1
                );
        }
        else if (ilayer == 0) { // 첫 번째 은닉 레이어인 경우
            for (i = 0; i < nhid_all[ilayer]; i++)
                activity(
                    input,
                    weights_opt[ilayer] + i * (n_model_inputs + 1),
                    hid_act[ilayer] + i,
                    n_model_inputs,
                    0
                );
        }
        else if (ilayer < n_all - 1) {  // 중간 위치의 은닉 레이어인 경우
            for (i = 0; i < nhid_all[ilayer]; i++)
                activity(
                    hid_act[ilayer - 1],
                    weights_opt[ilayer] + i * (nhid_all[ilayer - 1] + 1),
                    hid_act[ilayer] + i,
                    nhid_all[ilayer - 1],
                    0
                );
        }
        else {  // 출력 레이어인 경우
            for (i = 0; i < ntarg; i++)
                activity(
                    hid_act[ilayer - 1],
                    final_layer_weights + i * (nhid_all[ilayer - 1] + 1),
                    outputs + i,
                    nhid_all[ilayer - 1],
                    1
                );
        }
    }

    if (classifier) {   // 분류 목적이면 항상 Softmax 사용
        sum = 0.0;
        for (i = 0; i < ntarg; i++) {   // 모든 출력 대상 순회
            if (outputs[i] < 300.0) // Softmax가 너무 큰 값을 생성하는 것을 방지
                outputs[i] = exp(outputs[i]);
            else
                outputs[i] = exp(300.0);

            sum += outputs[i];
        }

        for (i = 0; i < ntarg; i++)
            outputs[i] /= sum;
    }
}

double batch_gradient (
    int istart,             // 입력 행렬의 첫 번째 데이터 인덱스
    int istop,              // 지난 마지막 데이터의 인덱스
    double *input,          // 입력 행렬; 각 데이터의 길이 == max_neurons
    double *targets,        // 목표(정답) 행렬; 각 데이터의 길이 == ntargs
    int n_all,              // 출력은 포함하고 입력은 제외한 레이어의 개수
    int n_all_weights,      // 마지막 레이어와 모든 바이어스 항을 포함한 총 가중치 개수
    int n_model_inputs,     // 모델 입력의 개수; 입력 행렬은 더 많은 열을 가질 수도 있음
    double *outputs,        // 모델의 출력 벡터; 여기서는 작업 벡터로 사용됨
    int ntarg,              // 출력의 개수
    int *nhid_all,          // nhid_all[i]는 i번째 은닉 레이어에 존재하는 뉴런의 개수
    double *weights_opt[],  // weights_opt[i]는 i번째 은닉 레이어의 가중치 벡터를 가리키는 포인터
    double *hid_act[],      // hid_act[i]는 i번째 은닉 레이어의 활성화 벡터를 가리키는 포인터
    int max_neurons,        // 입력 행렬의 열의 개수; n_model_inputs보다 최대치가 더 크다
    double *this_delta,     // 현재 레이어에 대한 델타 변수를 가리키는 포인터
    double *prior_delta,    // 다음 단계에 사용하기 위해 이전 레이에에서 미리 저장해놓은 델타 변수를 가리키는 포인터
    double **grad_ptr,      // grad_ptr[i]는 i번째 레이어의 기울기를 가리키는 포인터
    double *final_layer_weights,    // 마지막 레이어의 가중치를 가리키는 포인터
    double *grad,           // 계산된 모든 기울기로, 하나의 긴 벡터를 가리키는 포인터
    int classifier          // 0이 아니면 softmax로 출력하며, 1이면 선형 조합으로 출력
) {
    int i, j, icase, ilayer, nprev, nthis, nnext, imax;
    double diff, *dptr, error, *targ_ptr, *prevact, *gradptr, delta, *nextcoefs, tmax;

    for (i = 0; i < n_all_weights; i++) // 합산을 위해 기울기를 0으로 초기화
        grad[i] = 0.0;  // 모든 레이어의 gradient

    error = 0.0;        // 전체 오차 값 누적

    for (icase = istart; icase < istop; icase++) {
        dptr = input + icase * max_neurons; // 현재 데이터
        trial_thr(
            dptr,
            n_all,
            n_model_inputs,
            outputs,
            ntarg,
            nhid_all,
            weights_opt,
            hid_act,
            final_layer_weights,
            classifier
        );

        targ_ptr = targets + icase * ntarg;

        if (classifier) {   // Softmax를 사용한 경우
            tmax = -1.e30;

            for (i = 0; i < ntarg; i++) {   // 최댓값을 갖는 참 클래스를 찾는다.
                if (targ_ptr[i] > tmax) {
                    imax = i;
                    tmax = targ_ptr[i];
                }
                this_delta[i] = targ_ptr[i] - outputs[i];   // 교차 엔트로피를 입력(logit)으로 미분해 음의 부호를 취한 식
            }

            error -= log(outputs[imax] + 1.e-30);   // 음의 로그 확률을 최소화한다.
        }
        else {
            for (i = 0; i < ntarg; i++) {
                diff = outputs[i] - targ_ptr[i];
                error += diff * diff;
                this_delta[i] = -2.0 * diff;    // i번째 뉴런의 입력으로 제곱 오차를 미분해 음의 부호를 취한다.
            }
        }

        if (n_all == 1) {           // 은닉 레이어가 없는 경우
            nprev = n_model_inputs; // 출력 레이어에 전달되는 입력의 개수
            prevact = input + icase * max_neurons;  // 현재 데이터를 가리키는 포인터
        }
        else {
            nprev = nhid_all[n_all - 2];    // n_all-2 인덱스는 마지막 은닉 레이어
            prevact = hid_act[n_all - 2];   // 출력 레이어로 전달되는 레이어의 포인터 변수
        }
        
        gradptr = grad_ptr[n_all - 1];  // 기울기 벡터에서 출력 기울기를 가리키는 포인터

        for (i = 0; i < ntarg; i++) {   // 모든 출력 뉴런들에 대해 루프 연산 수행
            delta = this_delta[i];      // 평가 기준을 logit으로 편미분해 음수를 취한다.

            for (j = 0; j < nprev; j++)
                *gradptr++ += delta * prevact[j]; // 모든 훈련 데이터에 대한 결과를 누적한다.

            *gradptr++ += delta;    // 바이어스 활성화는 항상 1이다.
        }

        nnext = ntarg;  // 한 레이어 되돌아갈 준비를 한다.
        nextcoefs = final_layer_weights;

        for (ilayer = n_all - 2; ilayer >= 0; ilayer--) { // 각 은닉 레이어마다 역방향으로 진행한다.
            nthis = nhid_all[ilayer];   // 현재 은닉 레이어상에 존재하는 뉴런의 개수
            gradptr = grad_ptr[ilayer]; // 현재 레이어의 기울기를 가리키는 포인터

            for (i = 0; i < nthis; i++) {   // 현재 레이어 상의 뉴런들에 대해 루프 수행
                delta = 0.0;
                
                for (j = 0; j < nnext; j++)
                    delta += this_delta[j] * nextcoefs[j * (nthis + 1) + i];

                delta *= hid_act[ilayer][i] * (1.0 - hid_act[ilayer][i]);   // 미분 연산
                prior_delta[i] = delta; // 다음 레이어를 위해 저장

                if (ilayer == 0) {
                    prevact = input + icase * max_neurons;  // 현재 데이터를 가리키는 포인터

                    for (j = 0; j < n_model_inputs; j++)
                        *gradptr++ += delta * prevact[j];
                }
                else {  // 적어도 하나 이상의 은닉 레이어가 현재 레이어 이전에 존재
                    prevact = hid_act[ilayer - 1];

                    for (j = 0; j < nhid_all[ilayer - 1]; j++)
                        *gradptr++ += delta * prevact[j];
                }
                *gradptr++ += delta;    // 바이어스 활성화는 항상 1이다.
            }   // 현재 은닉 레이어상의 모든 뉴런을 대상으로 한다.

            for (i = 0; i < nthis; i++) // 현재 델타 값을 이전 델타 값으로 저장
                this_delta[i] = prior_delta[i];

            nnext = nhid_all[ilayer];   // 다음 레이어를 위한 준비
            nextcoefs = weights_opt[ilayer];
        }   // 모든 레이어를 대상으로 거꾸로 진행
    }       // 모든 데이터를 대상으로 순환 실행

    return error;   // MSE나 음의 로그 발생 가능 확률 반환
}

int main() {

    return 0;
}