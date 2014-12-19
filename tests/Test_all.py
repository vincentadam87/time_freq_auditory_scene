"""
Script to run all the tests
"""

import unittest


from tests.Test_Leaf import Test_Leaf
from tests.Test_Scene import Test_Scene
from tests.Test_Shepard import Test_Shepard

"""
Those tests only check input/output consistency
It does not check the content of the output!
"""

def main():
    test_suite = suite()
    runner = unittest.TextTestRunner()
    runner.run(test_suite)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(Test_Leaf())
    suite.addTest(Test_Scene())
    suite.addTest(Test_Shepard())

    return suite

if __name__ == '__main__':
    main()