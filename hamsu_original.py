# ==================================================================================================================
# 모듈 로딩
# ==================================================================================================================

# 샘플 데이터 관련 모듈
from sklearn.datasets import load_iris

# 데이터 준비 관련 모듈
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder    #데이테 라벨을 바꾸는거

# 학습 & 테스트용 데이터셋 준비 관련 모듈
from sklearn.model_selection import train_test_split

# 스케일링, 전처리 관련 모듈
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures

# 알고리즘 관련 모듈
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# 성능평가 관련 모듈
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# ==================================================================================================================
# 이상치 관련 함수들
# ==================================================================================================================

# 함수기능 : boxplot을 그리면서 이상치를 시각화로 확인하기
# 매개변수 : 사용할 데이터 프레임
# 반환 : boxplot에 대한 객체 반환

def visual_flier(insert_data):
    # 이상치 확인 by 시각화
    obj = plt.boxplot(insert_data)
    plt.title(f'box plot of data')
    plt.xlabel(insert_data.columns.tolist())
    plt.show()

    return obj


# ---------------------------------------------------------------------
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



# ---------------------------------------------------------------------
# 함수기능 : 이상치 계산하기 (q1, q3, iqr, lower, upper)
# 매개변수 : 사용할 데이터 프레임
# 반환 : lower, upper


def print_flier(insert_data):
    q1 = insert_data.quantile(0.25)
    q3 = insert_data.quantile(0.75)
    iqr = q3 - q1
    print(f"[ q1 ] \n{q1}\n\n[ q3 ] \n{q3}\n")
    print(f"[ iqr의 범위 ]\n{iqr}\n")

    print('------------------ 이상값이 될 기준 계산 ------------------\n')

    lower = q1 - 1.5 * iqr
    #print(f"[ lower의 값 ]\n{lower}\n")
    upper = q3 + 1.5 * iqr
    #print(f"[ upper의 값 ]\n{upper}\n")

    mask = insert_data < lower
    print(f"[ lower의 개수 ]\n{mask.sum()}\n")
    mask2 = insert_data > upper
    print(f"[ upper의 개수 ]\n{mask2.sum()}\n")

    return lower, upper


# ---------------------------------------------------------------------
# 함수기능 : 이상치 삭제하기
# 매개변수 : 사용할 데이터 프레임, 삭제할 컬럼명
# 반환 : 이상치를 삭제한 최종 데이터
# 주의 : 이상치 계산하는 거에서 upper, lower 반환값을 변수로 설정해두어야 함
# 미완성!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!실행안됨!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def drop_flier(insert_data, col_name):
    global lower
    global upper
    # 어쨰서 둘의 반환값이 같을까? 컬럼별로 적용하지 않아서 그런걸까?
    # data[~mask].shape
    # data[mask].shape
    # 이렇게 하면 데이터 프레임 전체에서 조건이 맞는걸 구해야하기 때문에 내가 의도한 바와 맞지 않는다.
    # 그래서 컬럼별로 필터링을 적용해주어야 한다.

    mask = insert_data[col_name] >= lower.loc[col_name]
    insert_data = insert_data[mask]
    print(f" '{col_name}' 컬럼에서 lower 미만의 값 제거 후 shape: {insert_data.shape}\n")

    mask2 = insert_data[col_name] <= upper.loc[col_name]
    insert_data = insert_data[mask2]
    print(f" '{col_name}' 컬럼에서 upper 초과의 값 제거 후 shape: {insert_data.shape}")

    return insert_data


# ---------------------------------------------------------------------
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


# <사용예시>
file = load_iris()
data = file['data']
data = pd.DataFrame(data, columns=['sepal-length', 'sepal-width', 'petal-length', 'petal-width'])
target = file['target']
target = pd.Series(target, name = 'variety')

visual_flier(data)

lower, upper = print_flier(data)

data = drop_flier(data, 'sepal-width')


# ==================================================================================================================
# 상관관계 관련 함수들
# ==================================================================================================================
# 함수기능 : 피처 간의 상관관계를 scatter로 시각화
# 매개변수 : 행수, 열수, 피쳐 리스트
# 반환 : 없음





def print_only_features(col_list):
    j = 1
    for i in range(len(col_list)):
        plt.figure(figsize=(15, 15))

        for k in range(len(col_list)):
            plt.subplot(len(col_list), len(col_list), j)
            plt.scatter(data[col_list[i]], data[col_list[k]])
            plt.xlabel(col_list[i])
            plt.ylabel(col_list[k])
            j += 1
        plt.tight_layout()
        plt.show()

#사용예시
# col_list = ['sepal-length', 'petal-length', 'petal-width']
# print_only_features(col_list)


# ---------------------------------------------------------------------
# 함수기능 : 피처와 타겟과의 관계 정도를 시각화 및 수치값으로 표기 및 출력
# 매개변수 : 행수, 열수, 타겟, 피쳐 리스트, 상관계수값
# 반환 : 없음

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

# 파라미터값 사용 예시
# nrows = 1
# ncols = 4
# df = use_data
# target = 'mpg'
# features = ['displacement', 'horsepower', 'weight', 'acceleration']


# ---------------------------------------------------------------------
# 함수기능 : 피처와 타겟과의 관계 정도를 시각화 및 수치값으로 표기 및 출력
# 매개변수 : 행수, 열수, 타겟, 피쳐 리스트
# 반환 : 없음

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