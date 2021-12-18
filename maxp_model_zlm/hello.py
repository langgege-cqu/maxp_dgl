# # class Solution(object):

# #     def __init__(self):
# #         self.memery = dict()
# #         self.memery[0] = 0
# #         self.memery[1] = 1

# #     def fib(self, n):
# #         """
# #         :type n: int
# #         :rtype: int
# #         """
# #         if n in self.memery.keys():
# #             return self.memery[n]
# #         else:
# #             result = self.fib(n - 2) + self.fib(n - 1)
# #             self.memery[n] = result
# #             return result

# # a = Solution()

# # print(a.fib(45))

# class Solution(object):

#     def findNumberIn2DArray(self, matrix, target):
#         """
#         :type matrix: List[List[int]]
#         :type target: int
#         :rtype: bool
#         """
#         if len(matrix) == 0:
#             return False
#         return self.findNumberInSub2DArray(matrix, 0, len(matrix) - 1, 0, len(matrix[0]) - 1, target)

#     def findNumberInSub2DArray(self, matrix, i_s, i_e, j_s, j_e, target):
#         if i_s > i_e or j_s > j_e:
#             return False

#         i_m = (i_s + i_e) // 2
#         j_m = (j_s + j_e) // 2

#         if matrix[i_m][j_m] == target:
#             return True
#         elif matrix[i_m][j_m] < target:
#             right = self.findNumberInSub2DArray(matrix, i_s, i_e, j_m + 1, j_e, target)
#             down = self.findNumberInSub2DArray(matrix, i_m + 1, i_e, j_s, j_m, target)
#             return right or down
#         else:
#             left = self.findNumberInSub2DArray(matrix, i_s, i_e, j_s, j_m - 1, target)
#             up = self.findNumberInSub2DArray(matrix, i_s, i_m - 1, j_m, j_e, target)
#             return left or up

# s = Solution()
# matrix = [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]]
# target = 5
# print(s.findNumberIn2DArray(matrix, target))

for i in range(2, 2):
    print(i)