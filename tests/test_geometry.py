"""Protect geometric comparison from silent coordinate and matching errors."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import inverse_boxes, match_stability

def pred(box,label=1):
    return {'box_base':box,'score':.8,'label_id':label}

class GeometryTests(unittest.TestCase):
    def test_horizontal_flip_edges(self):
        np.testing.assert_allclose(inverse_boxes([[70,5,90,20]],'hflip',100,60),[[10,5,30,20]])

    def test_ccw_rotation_on_nonsquare_image(self):
        np.testing.assert_allclose(inverse_boxes([[5,70,20,90]],'rot90',100,60),[[10,5,30,20]])

    def test_empty_prediction_is_not_perfect_stability(self):
        self.assertIsNone(match_stability([],[],100,60)['retained_fraction'])
        self.assertEqual(match_stability([pred([0,0,10,10])],[],100,60)['retained_fraction'],0)

    def test_duplicate_predictions_match_at_most_once(self):
        result=match_stability([pred([0,0,10,10]),pred([0,0,10,10])],[pred([0,0,10,10])],100,60)
        self.assertEqual(result['matched'],1)
        self.assertEqual(result['retained_fraction'],.5)

if __name__=='__main__':unittest.main()
