# --------------------------------------------------------------------
# 상속 (Inheritance)
# - 다중 상속 허용
# - 문법 : class 자식클래스명(부모클래스명, ...)
# ---------------------------------------------------------------------
class A:
    @classmethod
    def printInfo(cls):
        print('A')

class B:
    @classmethod
    def printInfo(cls):
        print('B')

class AB(A, B):       #A부터 가서 찾는다.
    pass

class CC(B, A):       #B부터 가서 찾는다.
    pass

ab1 = AB()
ab1.printInfo()
CC.printInfo()