import numpy as np
from sympy import *

c1, s1 = symbols("c1 s1")
c2, s2 = symbols("c2 s2")
c3, s3 = symbols("c3 s3")

r1 = np.array([[1, 0, 0],[0, c1, -s1],[0, s1, c1]]).reshape(3,3)
r2 = np.array([[c2, 0, s2],[0, 1, 0],[-s2, 0, c2]]).reshape(3,3)
r3 = np.array([[c3, -s3, 0],[s3, c3, 0],[0, 0, 1]]).reshape(3,3)

matrices = [r1, r2, r3]

dirs = ["x", "y", "z"]
for i in range(3):
  for j in range(3):
    if i != j:
      res = matrices[i]@matrices[j]
      print("// r{}_{} = ".format(i+1, j+1))

      print("template <typename T>")
      print(f"PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_{dirs[i]}{dirs[j]}(T ang{i+1}, T ang{j+1}, bool is_radian = true) {{")
      print(f"  if (!is_radian) {{ \n    ang{i+1} = degrees_to_radians(ang{i+1}); \n    ang{j+1} = degrees_to_radians(ang{j+1}); \n  }}")
      print(f"  T s{i+1} = sin(ang{i+1}); T s{j+1} = sin(ang{j+1});")
      print(f"  T c{i+1} = cos(ang{i+1}); T c{j+1} = cos(ang{j+1});")

      print("  return {{{}, {}, {}, {}, {}, {}, {}, {}, {}}};".format(*res.T.flatten()))
      print("}\n")

for i in range(3):
  for j in range(3):
    for k in range(3):
      if i != j and j != k:
        
        exec(f"c1{i+1}, s1{i+1} = symbols(\"c1{i+1} s1{i+1}\")")
        exec("c2{0}, s2{0} = symbols(\"c2{0} s2{0}\")".format(j+1))
        exec("c3{0}, s3{0} = symbols(\"c3{0} s3{0}\")".format(k+1))

        r1 = eval("np.array([[1, 0, 0],[0, c1{0}, -s1{0}],[0, s1{0}, c1{0}]]).reshape(3,3)".format(i+1))
        r2 = eval("np.array([[c2{0}, 0, s2{0}],[0, 1, 0],[-s2{0}, 0, c2{0}]]).reshape(3,3)".format(j+1))
        r3 = eval("np.array([[c3{0}, -s3{0}, 0],[s3{0}, c3{0}, 0],[0, 0, 1]]).reshape(3,3)".format(k+1))

        matrices = [r1, r2, r3]



        res = matrices[i]@matrices[j]@matrices[k]
        print("// r{}_{}_{} = ".format(i+1, j+1, k+1))

        print("template <typename T>")
        print(f"PE_HOST_DEVICE matrix_t<T, 3, 3> rotation_{dirs[i]}{dirs[j]}{dirs[k]}(T ang1, T ang2, T ang3, bool is_radian = true) {{")
        print(f"  if (!is_radian) {{ \n" + 
               f"    ang{i+1} = degrees_to_radians(ang{i+1}); \n" + 
               f"    ang{j+1} = degrees_to_radians(ang{j+1}); \n" +
               f"    ang{k+1} = degrees_to_radians(ang{k+1}); \n" +
               "  }")
        
        print(f"  T s1{i+1} = sin(ang1); T s2{j+1} = sin(ang2); T s3{k+1} = sin(ang3);")
        print(f"  T c1{i+1} = cos(ang1); T c2{j+1} = cos(ang2); T c3{k+1} = cos(ang3);")

        print("  return {{{}, {}, {}, {}, {}, {}, {}, {}, {}}};".format(*res.T.flatten()))
        print("}\n")
