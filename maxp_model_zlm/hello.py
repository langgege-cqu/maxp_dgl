class Solution(object):

    def __init__(self):
        self.memery = dict()
        self.memery[0] = 0
        self.memery[1] = 1

    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n in self.memery.keys():
            return self.memery[n]
        else:
            result = self.fib(n - 2) + self.fib(n - 1)
            self.memery[n] = result
            return result


a = Solution()

print(a.fib(45))
