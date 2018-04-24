import unittest
import imdecoder

class TestImageHandlingMethods(unittest.TestCase):
    def test_loadable(self):
        img = imdecoder.load_img('./assets/testimg02')
        self.assertTrue(img is not None)
    def test_notloadable(self):
        img = imdecoder.load_img('./assets/test')
        self.assertTrue(img is None)

class TestImageProcessingMethods(unittest.TestCase):
    def setUp(self):
        imgs = [imdecoder.load_img('./assets/testimg0%d'% (n)) for n in xrange(1,13)]
        self.detectors = [imdecoder.Detector(img) for img in imgs]

    def test_shapes_detected(self):
        for d in self.detectors:
            self.assertGreater(d.num_of_objects ,2)
    
    def test_3_anchors(self):
        for idx in xrange(len(self.detectors)):
            an = self.detectors[idx].num_of_anchors
            self.assertEqual(an, 3,
                    "image %d expected 3 anchors %d found"%(idx+1, an))

    def test_16_bin(self):
        for idx in xrange(len(self.detectors)):
            an = self.detectors[idx].num_of_bins
            self.assertEqual(an, 16,
                    "image %d expected 16 bin %d found"%(idx+1, an))
        
    def test_shapes_19(self):
        for idx in xrange(len(self.detectors)):
            l = self.detectors[idx].num_of_objects
            self.assertEqual(l, 19,
                    "image %d expected 19 output %d"%(idx+1,l) )

if __name__ == '__main__':
    unittest.main()
