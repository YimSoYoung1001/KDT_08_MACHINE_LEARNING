## ------------------------------------------------------------------------------
## 모델을 활용한 서비스 제공
## ------------------------------------------------------------------------------

# 모듈 로딩
from joblib import load

# 전역 변수
model_file = '../model/iris_df.pkl'
# 백 슬러시가 아니다 !

# 모델 로딩
model = load(model_file)

# 로딩된 모델 확인
print(model.classes_)

# 붓꽃 정보 입력    => 4개 피처
datas = input('붓꽃 정보 입력 (예: 꽃받침길이, 꽃받침너비, 꽃잎 길이, 꽃잎 너비 순서): ')
if len(datas):
    datas_list = list(map(float, datas.split(',')))       # map object가 return 되니까 list로 형변환 해야함
    print(datas_list,'\n')

    # 입력된 정보에 해당하는 품종 알려주기
    # 모델의 predict(2D)
    pre_iris = model.predict([datas_list])
    proba = model.predict_proba([datas_list])[0].max()

    print(f"해당 꽃은 {proba}% {pre_iris}입니다.")

else:
    print('입력된 정보가 없습니다.')