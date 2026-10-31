import itertools
import unittest
import hashlib
from pathlib import Path
import tempfile
from tensorfold.tp4.cache_repartition import groups,extract,target,cells,bounds
from tensorfold.tp4.cache_repartition import publish_rank


class Repartition(unittest.TestCase):
    def test_verified_export_is_discoverable_by_the_native_prefix_contract(self):
        root = Path(__file__).resolve().parents[1]/'local-results'
        root.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='400-rank-name-', dir=root))
        pending = directory/'rank2of4.partial'
        pending.write_bytes(b'exact-cache-bytes')
        target_file = directory/'rank2of4.bin'
        digest = hashlib.sha256(pending.read_bytes()).hexdigest()
        publish_rank(pending, target_file, digest)
        found = list(directory.glob('rank2of4-*.bin'))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, f'rank2of4-{digest[:20]}.bin')
        self.assertTrue(found[0].samefile(target_file))
        self.assertFalse(found[0].is_symlink())

    def test_rectangles_keep_all_exact_bytes_with_unequal_both_axis_cuts(self):
        shape=(6,8,2);raw=bytes(range(96));indexes=[];maps=[]
        for rank,(lo,hi) in enumerate([(0,2),(2,4),(4,6)]):
            indexes.append([[f'x|(0,128)|({lo*16},{hi*16})|tr','uint8',[hi-lo,8,2],0,(hi-lo)*16]])
            maps.append(raw[lo*16:hi*16])
        g=groups(indexes)[('x.trellis','None')]
        for a,b in [(0,3),(3,6)]:
            for c,d in [(0,5),(5,8)]:
                expected=bytes(raw[(i*8+j)*2+k] for i,j,k in itertools.product(range(a,b),range(c,d),range(2)))
                self.assertEqual(extract([(a,b),(c,d),(0,2)],g,maps),expected)
                self.assertEqual(extract([(a,b),(c,d),(0,2)],g,maps,True),expected)

    def test_replication_is_copied_once(self):
        ix=[[['norm|None','float16',[4],0,8]]]*3
        self.assertEqual(extract([(0,4)],groups(ix)[('norm','None')],[b'abcdefgh']*3),b'abcdefgh')
        self.assertEqual(extract([(0,4)],groups(ix)[('norm','None')],[b'abcdefgh']*3,True),b'abcdefgh')

    def test_gap_rejected(self):
        with self.assertRaises(ValueError):list(cells([(0,5)],[dict(bounds=[(0,2)]),dict(bounds=[(3,5)])]))

    def test_unpadding_discards_tail_without_requantization(self):
        class Op:axis=0;count=2
        g=dict(shape=[8],dtype='float16',sources=[dict(rank=0,bounds=[(0,8)],shape=[8],offset=0)])
        wanted=target('embed.weight|None',g,lambda n:Op)
        self.assertEqual(wanted,[(0,6)])
        self.assertEqual(extract(wanted,g,[b'abcdefghijklmnop']),b'abcdefghijkl')
        self.assertEqual(extract(wanted,g,[b'abcdefghijklmnop'],True),b'abcdefghijkl')

    def test_unaligned_or_missing_geometry_rejected(self):
        with self.assertRaises(ValueError):bounds('x|(0,129)|None|tr',[2,8,2])
        with self.assertRaises(ValueError):target('x|(0,256)|None|tr',dict(shape=[2,8,2]),lambda n:None)

    def test_scalar(self):
        g=groups([[['epsilon|None','float32',[],0,4]]])[('epsilon','None')]
        self.assertEqual(extract([],g,[b'abcd']),b'abcd')


if __name__=='__main__':unittest.main()
