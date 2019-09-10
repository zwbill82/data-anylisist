import unittest
import pandas as pd
from time_sequeces_learn import  TimeSeuece

class MyTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()
        data = pd.read_csv("../data/AirPassengers.csv", parse_dates=['Month'], index_col='Month')
        print(data.head())
        ts = data["#Passengers"]
        self.tsq=TimeSeuece(ts)


    def test_expwighted_smooth(self):
        self.tsq.expwight_smooth()

    def test_diff_smooth(self):
        self.tsq.diff_smooth()


if __name__ == '__main__':
    unittest.main()
