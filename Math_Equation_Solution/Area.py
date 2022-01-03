
# Area of Rectangle

def area_rect(a,b):
    area_rectangle = a * b
    print("Area of rectangle = A * B where A and B are sides")
    return area_rectangle
side1 = int(input("A : "))
side2 = int(input("B : "))
print(area_rect(side1,side2))

def area_square(a):
    area_sq = a * a
    print("Area of Square = A * A where A is the side of Square")
    return area_sq
print(area_square(side1))

def area_parllgm(b,h):
    area_pargm = b*h
    print("Area of parallelogram = B * H where B is the base and H is the height")
    return area_pargm
base = int(input("Base:"))
height = int(input("Height :"))
print(area_parllgm(base,height))

def area_trapezoid(a,b,h):
    area_trape = a * b* h
    print("Area of Trapezium = A * B * H where A , B area base and H is the height")
    return area_trape
base1 = int(input("Base1:"))
print(area_trapezoid(base1,base,height))

def area_rhombus(d1,d2):
    area_rhms = (d1*d2)/2
    print("Area of Rhombus = (D1 * D2 )/2 where D1 and D2 are diagonels")
    return area_rhms
d1 = int(input("Diagonel1:"))
d2 = int(input("Diagonel2:"))
print(area_rhombus(d1,d2))

# Area of circle

def area_circle(r):
    area_cir = (22/7)*(r*r)
    print("Area of circle = pi * r*r , where r is the radius")
    return area_cir
r = int(input("Radius of circle:"))
print(area_circle(r))

