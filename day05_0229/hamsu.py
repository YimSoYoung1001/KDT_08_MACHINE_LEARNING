import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===================================================================================================================
# 함수기능 : 피처와 타겟과의 관계 정도를 시각화 및 수치값으로 표기 및 출력
# 매개변수 : 행수, 열수, 타겟, 피쳐 리스트, 상관계수값
# 반환 : 없음

# 파라미터값 사용 예시
# nrows = 1
# ncols = 4
# df = use_data
# target = 'mpg'
# features = ['displacement', 'horsepower', 'weight', 'acceleration']

def print_feature(nrows, ncols, df, target, features, corr_list):
    # idx를 결정하는 것은 피쳐의 수다.
    for idx in range(len(features)):
        # 인덱스는 0번 부터 시작하니까 +1 해주어야 함
        plt.subplot(nrows, ncols, idx + 1)

        # 타겟과 나머지 피쳐들이 for문 돌면서 순서대로 들어감
        # corr은 라벨을 설정하기 위해서 넣어줌 (범례)
        plt.scatter(df[target], df[features[idx]], label=f"{corr_list[idx]:.2}")

        plt.legend()
        plt.xlabel(target)
        plt.ylabel(features[idx])
    plt.tight_layout()  # 표들끼리 겹치지 않게끔 해준다
    plt.show()


def print_feature_corrNone(nrows, ncols, df, target, features):
    # idx를 결정하는 것은 피쳐의 수다.
    for idx in range(len(features)):
        # 인덱스는 0번 부터 시작하니까 +1 해주어야 함
        plt.subplot(nrows, ncols, idx + 1)

        # 타겟과 나머지 피쳐들이 for문 돌면서 순서대로 들어감
        # corr은 라벨을 설정하기 위해서 넣어줌 (범례)
        plt.scatter(df[target], df[features[idx]])

        plt.legend()
        plt.xlabel(target)
        plt.ylabel(features[idx])
    plt.tight_layout()  # 표들끼리 겹치지 않게끔 해준다
    plt.show()

# ===================================================================================================================
# 함수기능 : 컬럼에 대한 boxplot 그려서 이상치 확인하기
# 매개변수 : 컬럼리스트, 데이터
# 반환 : 없음

# 파라미터값 사용 예시
# col_list =['mpg', 'cylinders', 'displacement']
# use_data = data
def draw_boxplot(col_list, use_data):
    bp_obj = []
    for i in range(len(col_list)):
        plt.subplot(1 ,len(col_list), i+1)
        plt.xlabel(col_list[i], fontsize=15)
        obj = plt.boxplot(use_data[col_list[i]])
        bp_obj.append(obj)
    plt.tight_layout()  # 표들끼리 겹치지 않게끔 해준다
    plt.show()
    #return bp_obj

    for i in range(len(bp_obj)):
        value = bp_obj[i]
        # 각 컬럼별로 이상치인 값들이 나온다.
        print(value['fliers'][0].get_ydata())




# ===================================================================================================================
# 함수기능 : 컬럼에 대한 boxplot 그려서 이상치 확인하기
# 매개변수 : 스케일링 알고리즘 객체
# 반환 : scaler, scaled_X_train, scaled_X_test, model

# 파라미터값 사용 예시
#object = StandardScaler()
#scaling(object)



X_train = 0
X_test = 0
y_train = 0
y_test = 0

test_dict = {}
def scaling(scale_type):
    scaler = scale_type
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(scaled_X_train, y_train)
    print(f"[모델 파라미터] \n - coef_ : {model.coef_} \n - intercept_ : {model.intercept_}\n")

    train_score = model.score(scaled_X_train, y_train)
    test_score = model.score(scaled_X_test, y_test)
    print(f"[score 비교] \n  train_score = {train_score} --- test_score = {test_score}")
    print()
    test_dict[scale_type] = test_score

    return scaler, scaled_X_train, scaled_X_test, model


# ===================================================================================================================
# 함수기능 : 성능평가
# 매개변수 :
# 반환 : 없음

'''
y_pred = model.predict(scaled_X_test)
y_pred

print('[ 모델설명도 ]')
print(f"설정계수값(R2) : {r2_score(y_test, y_pred)}")

# 처음에 테스트 데이터셋이 아닌 >>피쳐 데이터프레임 전체<<를 넣어버림! => 그래서 R2의 값이 마이너스의 값이 나옴 (잘못되었다는 예기)
# 그런데 R2의 값은 0~1 범위의 값이 나온다. 마이너스가 나왔다는 것은 무언가 잘못되었다는 이야기

print('[ 에러 ]')
print(f"평균제곱오차(MSE) : {mean_squared_error(y_test, y_pred)}")
print(f"평균절대값오차(MAE) : {mean_absolute_error(y_test, y_pred)}")
'''