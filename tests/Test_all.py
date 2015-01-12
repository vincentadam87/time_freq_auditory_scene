"""
Script to run all the tests
"""
import os, sys
import unittest

# add parent folder to path ( ../tests )
current_folder = os.path.abspath('.')
parent_folder = os.path.split(current_folder)[0]
sys.path.append(current_folder)
sys.path.append(parent_folder)


from Test_Scene import Test_Scene
from Test_Leaf import Test_Leaf
from Test_Shepard import Test_Shepard
from Test_Chambers import Test_Chambers
from Test_Nodes import Test_Nodes
from Test_SceneDrawer import Test_SceneDrawer

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
    suite.addTest(Test_Chambers())
    suite.addTest(Test_Nodes())
    suite.addTest(Test_SceneDrawer())

    return suite

if __name__ == '__main__':
    main()