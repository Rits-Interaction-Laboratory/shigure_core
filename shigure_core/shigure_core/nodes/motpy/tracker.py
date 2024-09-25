import time
import uuid
from collections.abc import Iterable
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import numpy as np
import scipy
from filterpy.kalman import KalmanFilter

import inspect
strongsort_path = inspect.getfile(KalmanFilter)


from loguru import logger
logger.info(strongsort_path)
from motpy.core import Box, Detection, Track, Vector, setup_logger
from motpy.metrics import angular_similarity, calculate_iou
from motpy.model import Model, ModelPreset
import itertools
logger = setup_logger(__name__)


def get_kalman_object_tracker(model: Model,box: Optional[Box] = None,score: Optional[float] = None, x0: Optional[Vector] = None) -> KalmanFilter:


    """ returns Kalman-based tracker based on a specified motion model spec.
        e.g. for spec = {'order_pos': 1, 'dim_pos': 2, 'order_size': 0, 'dim_size': 1}
        we expect the following setup:
        state x, x', y, y', w, h
        where x and y are centers of boxes
              w and h are width and height
    """
    # print(len(box[4])*2)

    tracker = KalmanFilter(dim_x=model.state_length+(len(box[4])*2),
                           dim_z=model.measurement_length)
    # print(model.state_length+(len(box[4])*2))
#    logger.info(box)
#    logger.info(x0)
    # print("x0")
    # print(x0)
    tracker.P = model.build_P(box)
    tracker.F = model.build_F(box,x0)
    tracker.Q = model.build_Q(box)
    tracker.H = model.build_H(box)
    tracker.R = model.build_R(box,x0,score)
#    tracker.P = model.build_P()

    if x0 is not None:
        tracker.x = x0
    #logger.info(tracker)
    #logger.info(tracker.x)
    # print(tracker)

    return tracker


DEFAULT_MODEL_SPEC = ModelPreset.constant_velocity_and_static_box_size_2d.value


def exponential_moving_average_fn(gamma: float) -> Callable:
    def fn(old, new):
        if new is None:
            return old

        if isinstance(new, Iterable):
            new = np.array(new)

        if old is None:
            return new  # first call

        if isinstance(old, Iterable):
            old = np.array(old)

        return gamma * old + (1 - gamma) * new

    return fn


class SingleObjectTracker:
    def __init__(self,
                 max_staleness: float = 12.0,
                 smooth_score_gamma: float = 0.8,
                 smooth_feature_gamma: float = 0.9,
                 score0: Optional[float] = None,
                 class_id0: Optional[int] = None):
        self.id: str = str(uuid.uuid4())
        self.steps_alive: int = 1
        self.steps_positive: int = 1
        self.staleness: float = 0.0
        self.max_staleness: float = max_staleness

        self.update_score_fn: Callable = exponential_moving_average_fn(smooth_score_gamma)
        self.update_feature_fn: Callable = exponential_moving_average_fn(smooth_feature_gamma)

        self.score: Optional[float] = score0
        self.feature: Optional[Vector] = None

        self.class_id_counts: Dict = dict()
        self.class_id: Optional[int] = self.update_class_id(class_id0)

        logger.debug(f'creating new tracker {self.id}')

    def box(self) -> Box:
        raise NotImplementedError()

    def is_invalid(self) -> bool:
        raise NotImplementedError()

    def _predict(self) -> None:
        raise NotImplementedError()

    def predict(self) -> None:
        self._predict()
        self.steps_alive += 1

    def update_class_id(self, class_id: Optional[int]) -> Optional[int]:
        """ find most frequent prediction of class_id in recent K class_ids """
        if class_id is None:
            return None

        if class_id in self.class_id_counts:
            self.class_id_counts[class_id] += 1
        else:
            self.class_id_counts[class_id] = 1

        return max(self.class_id_counts, key=self.class_id_counts.get)

    def _update_box(self, detection: Detection) -> None:
        raise NotImplementedError()

    def update(self, detection: Detection) -> None:
        self._update_box(detection)

        self.steps_positive += 1

        self.class_id = self.update_class_id(detection.class_id)
        self.score = self.update_score_fn(old=self.score, new=detection.score)
        self.feature = self.update_feature_fn(old=self.feature, new=detection.feature)



        # reduce the staleness of a tracker, faster than growth rate
        self.unstale(rate=3)

    def stale(self, rate: float = 1.0) -> float:
        self.staleness += rate
        return self.staleness

    def unstale(self, rate: float = 2.0) -> float:
        self.staleness = max(0, self.staleness - rate)
        return self.staleness

    def is_stale(self) -> bool:
        return self.staleness >= self.max_staleness

    def __repr__(self) -> str:
        return f'(box: {str(self.box())}, score: {self.score}, class_id: {self.class_id}, staleness: {self.staleness:.2f})'


class KalmanTracker(SingleObjectTracker):
    """ A single object tracker using Kalman filter with specified motion model specification """

    def __init__(self,
                 model_kwargs: dict = DEFAULT_MODEL_SPEC,
                 x0: Optional[Vector] = None,
                 box0: Optional[Box] = None,
                 score0: Optional[float] = None,
                 **kwargs) -> None:

        super(KalmanTracker, self).__init__(**kwargs)

        self.model_kwargs: dict = model_kwargs
        self.model = Model(**self.model_kwargs)

        if x0 is None:
            x0 = self.model.box_to_x(box0)

        self._tracker: KalmanFilter = get_kalman_object_tracker(model=self.model,box=box0,score=score0,x0=x0)

    def _predict(self) -> None:
        self._tracker.predict()
        #print(self._tracker)

    def _update_box(self, detection: Detection) -> None:
        z = self.model.box_to_z(detection.box)
        self._tracker.update(z)

    def box(self) -> Box:
        return self.model.x_to_box(self._tracker.x)

    def is_invalid(self) -> bool:
        try:
            has_nans = any(np.isnan(self._tracker.x))
            return has_nans
        except Exception as e:
            logger.warning(f'invalid tracker - exception: {e}')
            return True


class SimpleTracker(SingleObjectTracker):
    """ A simple single tracker with no motion modeling and box update using exponential moving averege """

    def __init__(self,
                 box0: Optional[Box] = None,
                 box_update_gamma: float = 0.5,
                 **kwargs):

        super(SimpleTracker, self).__init__(**kwargs)
        self._box: Box = box0

        self.update_box_fn: Callable = exponential_moving_average_fn(box_update_gamma)

    def _predict(self) -> None:
        pass

    def _update_box(self, detection: Detection) -> None:
        self._box = self.update_box_fn(old=self._box, new=detection.box)

    def box(self) -> Box:
        return self._box

    def is_invalid(self) -> bool:
        try:
            return any(np.isnan(self._box))
        except Exception as e:
            logger.warning(f'invalid tracker - exception: {e}')
            return True


""" assignment cost calculation & matching methods """


def _sequence_has_none(seq: Sequence[Any]) -> bool:
    return any([r is None for r in seq])


def cost_matrix_iou_feature(trackers: Sequence[SingleObjectTracker],
                            detections: Sequence[Detection],
                            feature_similarity_fn=angular_similarity,
                            feature_similarity_beta: float = None) -> Tuple[np.ndarray, np.ndarray]:

    # boxes
    # print("ni")
    # print(trackers)
    b1 = np.array([t.box()[0:4] for t in trackers])
    # print(b1)
    # b1 = [b1[0][0:4]]
    # print(b1)
    #print(b1[0][0:4])
    flat_list = []
    # for d in detections:
    #     print(flatten(d.box)[0:6])
    b2 = np.array([d.box[0:4] for d in detections])
    # print(b2)

    # box iou
    inferred_dim = int(len(b1[0]) / 2)
    iou_mat = calculate_iou(b1, b2, dim=inferred_dim)
    # print(iou_mat)

    # feature similarity
    if feature_similarity_beta is not None:
        # get features
        f1 = [t.feature for t in trackers]
        f2 = [d.feature for d in detections]

        if _sequence_has_none(f1) or _sequence_has_none(f2):
            # fallback to pure IOU due to missing features
            apt_mat = iou_mat
        else:
            sim_mat = feature_similarity_fn(f1, f2)
            sim_mat = feature_similarity_beta + (1 - feature_similarity_beta) * sim_mat

            # combined aptitude
            apt_mat = np.multiply(iou_mat, sim_mat)
    else:
        apt_mat = iou_mat

    cost_mat = -1.0 * apt_mat
    #print(cost_mat)
    return cost_mat, iou_mat

def flatten_sequences(sequences) -> list:
    sequences = [i if type(i) == list else [i] for i in sequences]
    flattened = list(itertools.chain.from_iterable(sequences))
    return flattened
def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, (list,tuple)):
            flat_list.extend(flatten(item))  # 再帰呼び出し
        else:
            flat_list.append(item)
    return flat_list


EPS = 1e-7


def match_by_cost_matrix(trackers: Sequence[SingleObjectTracker],
                         detections: Sequence[Detection],
                         min_iou: float = 0.1,
                         multi_match_min_iou: float = 1. + EPS,
                         **kwargs) -> np.ndarray:
    if len(trackers) == 0 or len(detections) == 0 :
        return []
    b1 = [d.box[0:4] for d in detections]
    flag = b1[0][0] == None
    if flag:
        # print("asdS")
        return []

    cost_mat, iou_mat = cost_matrix_iou_feature(trackers, detections, **kwargs)
    # print(cost_mat)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)

    matches = []
    for r, c in zip(row_ind, col_ind):
        # check linear assignment winner
        if iou_mat[r, c] >= min_iou:
#            logger.info(iou_mat[r, c])
            matches.append((r, c))

        # check other high IOU detections
        if multi_match_min_iou < 1.:
            for c2 in range(iou_mat.shape[1]):
               # logger.info(iou_mat[r, c2])
                if c2 != c and iou_mat[r, c2] > multi_match_min_iou:
                    #logger.info(iou_mat[r, c2])
                    matches.append((r, c2))

    return np.array(matches)


class BaseMatchingFunction:
    def __call__(self,
                 trackers: Sequence[SingleObjectTracker],
                 detections: Sequence[Detection]) -> np.ndarray:
        raise NotImplementedError()


class IOUAndFeatureMatchingFunction(BaseMatchingFunction):
    """ class implements the basic matching function, taking into account
    detection boxes overlap measured using IOU metric and optional 
    feature similarity measured with a specified metric """

    def __init__(self, min_iou: float = 0.1,
                 multi_match_min_iou: float = 1. + EPS,
                 feature_similarity_fn: Callable = angular_similarity,
                 feature_similarity_beta: Optional[float] = None) -> None:
        self.min_iou = min_iou
        self.multi_match_min_iou = multi_match_min_iou
        self.feature_similarity_fn = feature_similarity_fn
        self.feature_similarity_beta = feature_similarity_beta

    def __call__(self,
                 trackers: Sequence[SingleObjectTracker],
                 detections: Sequence[Detection]) -> np.ndarray:
        
        return match_by_cost_matrix(
            trackers, detections,
            min_iou=self.min_iou,
            multi_match_min_iou=self.multi_match_min_iou,
            feature_similarity_fn=self.feature_similarity_fn,
            feature_similarity_beta=self.feature_similarity_beta)


class MultiObjectTracker:
    def __init__(self, dt: float,
                 model_spec: Union[str, Dict] = DEFAULT_MODEL_SPEC,
                 matching_fn: Optional[BaseMatchingFunction] = None,
                 tracker_kwargs: Dict = None,
                 matching_fn_kwargs: Dict = None,
                 active_tracks_kwargs: Dict = None) -> None:
        """
            model_spec specifies the dimension and order for position and size of the object
            matching_fn determines the strategy on which the trackers and detections are assigned.

            tracker_kwargs are passed to each single object tracker
            active_tracks_kwargs limits surfacing of fresh/fading out tracks
        """

        self.trackers: List[SingleObjectTracker] = []
        self.frame = 0
        self.t_id = 0

        # kwargs to be passed to each single object tracker
        self.tracker_kwargs: Dict = tracker_kwargs if tracker_kwargs is not None else {}
        self.tracker_clss: Optional[Type[SingleObjectTracker]] = None

        # translate model specification into single object tracker to be used
        if model_spec is None:
            self.tracker_clss = SimpleTracker
            if dt is not None:
                logger.warning('specified dt is ignored in simple tracker mode')
        elif isinstance(model_spec, dict):
            self.tracker_clss = KalmanTracker
            self.tracker_kwargs['model_kwargs'] = model_spec
            self.tracker_kwargs['model_kwargs']['dt'] = dt
        elif isinstance(model_spec, str) and model_spec in ModelPreset.__members__:
            self.tracker_clss = KalmanTracker
            self.tracker_kwargs['model_kwargs'] = ModelPreset[model_spec].value
            self.tracker_kwargs['model_kwargs']['dt'] = dt
        else:
            raise NotImplementedError(f'unsupported motion model {model_spec}')

        logger.debug(f'using single tracker of class: {self.tracker_clss} with kwargs: {self.tracker_kwargs}')

        self.matching_fn: BaseMatchingFunction = matching_fn
        self.matching_fn_kwargs: Dict = matching_fn_kwargs if matching_fn_kwargs is not None else {}
        if self.matching_fn is None:
            self.matching_fn = IOUAndFeatureMatchingFunction(**self.matching_fn_kwargs)

        # kwargs to be used when self.step returns active tracks
        self.active_tracks_kwargs: Dict = active_tracks_kwargs if active_tracks_kwargs is not None else {}
        logger.debug('using active_tracks_kwargs: %s' % str(self.active_tracks_kwargs))

        self.detections_matched_ids = []

    def active_tracks(self,
                      max_staleness_to_positive_ratio: float = 3.0,
                      max_staleness: float = 999,
                      min_steps_alive: int = -1) -> List[Track]:
        """ returns all active tracks after optional filtering by tracker steps count and staleness """

        tracks: List[Track] = []
        print(len(self.trackers))
        for tracker in self.trackers:
            cond1 = tracker.staleness / tracker.steps_positive < max_staleness_to_positive_ratio  # early stage
            cond2 = tracker.staleness < max_staleness
            # cond3 = tracker.steps_alive >= min_steps_alive
            if cond1 and cond2 :#and cond3:
                tracks.append(Track(id=tracker.id, box=tracker.box(), score=tracker.score, class_id=tracker.class_id))
            # tracks.append(Track(id=tracker.id, box=tracker.box(), score=tracker.score, class_id=tracker.class_id))


        logger.debug('active/all tracks: %d/%d' % (len(self.trackers), len(tracks)))
        return tracks

    def cleanup_trackers(self) -> None:
        count_before = len(self.trackers)
        self.trackers = [t for t in self.trackers if not (t.is_stale() or t.is_invalid())]
        count_after = len(self.trackers)
        logger.debug('deleted %s/%s trackers' % (count_before - count_after, count_before))

    def step(self, detections: Sequence[Detection],f_matchs) -> List[Track]:
        """ the method matches the new detections with existing trackers,
        creates new trackers if necessary and performs the cleanup.
        Returns the active tracks after active filtering applied """
        t0 = time.time()
        print("f_match")
        print(f_matchs)

        # filter out empty detections
        #print(detections )
        detections = [det for det in detections if det is not None or det.feature is not None]

        for det in detections:
            print("dwee")
            if det.box[0] is None:
                print("ssd")
                for t in self.trackers:
                    x = det.box[4][0][0] - t.box()[4]
                    y = det.box[4][0][1] - t.box()[5]
                    # x1 =det.box[4][1][0] - t.box()[6]
                    # y1 = det.box[4][1][1] - t.box()[7]

                    # x2 =det.box[4][2][0] - t.box()[8]
                    # y2 = det.box[4][2][1] - t.box()[9]

                    # x3 =det.box[4][3][0] - t.box()[10]
                    # y3 = det.box[4][3][1] - t.box()[11]

                    det.box[0] = x #(x+x1+x2+x3)/4
                    det.box[1] = y #(y+y1+y2+y3)/4


                    det.box[2] = t.box()[2] 
                    det.box[3] = t.box()[3]

                    # det.box[4][0][0] = det.box[4][0][0] - x
                    # det.box[4][0][1] = det.box[4][0][1] -y
                    # det.box[4][1][0] = det.box[4][1][0] - x
                    # det.box[4][1][1] = det.box[4][1][1] - y

                    # det.box[4][2][0] =det.box[4][2][0] - x
                    # det.box[4][2][1] =det.box[4][2][0] - y

                    # x3 =det.box[4][3][0] - t.box()[10]
                    # y3 = det.box[4][3][1] - t.box()[11]
                    # det.box[3] = t.box()[3]
                    # det.box[2] = t.box()[2] 
                    # det.box[3] = t.box()[3]
                    # det.box[2] = t.box()[2] 
                    # det.box[3] = t.box()[3]


                


            # print(det.box)
            print(det.class_id)



        # print(det.box[0])


        # for t in self.trackers:
        #     print("trara")
        #     print(t.box()[0])
        #     print(t.box()[1])
        #     print(t.box()[2])
        #     print(t.box()[3])
        #     print(t.box()[4])

        # print(detections )
        

        # flow_detection = [flow_det for flow_detin ]
        
        # print(self.trackers)
        t_test = []
        bcount = 0

        # predict state in all trackers
        for t in self.trackers:
            
            # print(t)
            t.predict()
            if t.class_id == "stuffed toy":
                self.t_id = bcount
            print(t.box()[0])
            bcount += 1

        print('eeeeeeeeee')

        print(self.trackers)
        # print( self.t_id )




        # for det in detections:
        #     print(det.box)
        #     if det.box[0] == None:
        #         if len(t_test)>0 and len(det.box[4])== 4:
        #             print(t_test[4])
        #             x =det.box[4][0][0] - t_test[4]
        #             y = det.box[4][0][1] - t_test[5]

        #             x1 =det.box[4][1][0] - t_test[6]
        #             y1 = det.box[4][1][1] - t_test[7]

        #             x2 =det.box[4][2][0] - t_test[8]
        #             y2 = det.box[4][2][1] - t_test[9]

        #             x3 =det.box[4][3][0] - t_test[10]
        #             y3 = det.box[4][3][1] - t_test[11]

        #             p_x = (x +x1+x2+x3)/4
        #             p_y = (y+y1+y2+y3)/4

        #             det.box[0]= p_x
        #             det.box[1]= p_y

        #             det.box[2]= t_test[2]
        #             det.box[3]= t_test[3]


        #             print(det.box[4][0][0] - t_test[4])

            # if det.box[0] == None:
                # pre = self.trackers.box[4:]
                # print
                # det.box[5]

        

        # match trackers with detections
        logger.debug('step with %d detections' % len(detections))

#        logger.info(self.trackers)
        matches = self.matching_fn(self.trackers, detections)
        
        logger.debug('matched %d pairs' % len(matches))

        self.detections_matched_ids = [None] * len(detections)
#        logger.info(matches)
        
        
    #     if len(matches) >= 0 and len(f_matchs) > 0 and len(self.trackers) >0 :
    #         print("sdddd")
    #         f_id = f_matchs.values()
    #         for key in f_id :
    #             det_idx = int(key)-1

    #             # det = detections[det_idx]
    # #            logger.info(det.score)
    #             # tracker = self.tracker_clss(box0=det.box,
    #             #                             score0=det.score,
    #             #                             class_id0=det.class_id,
    #             #                             **self.tracker_kwargs)
                
    #             self.detections_matched_ids[det_idx] = self.trackers[det_idx].id
    #             self.trackers[int(key)-1].update(detection=detections[det_idx])
    #     else:


        # assigned trackers: correct
        print("match")
        print(matches)
        list_t = []
        f_id = f_matchs.values()
        for match ,key in zip(matches, f_id):
            
            track_idx, det_idx = match[0], match[1]

            # print("ffff")
            # # print(int(key)-1)
            # print(track_idx)
            # print(det_idx)

            if 'or' in key:
                key = key.split()[0]
                match[0] = int(key)

            else:
                match[0] = int(key)




            track_idx2 = int(key)

            # match[0] = int(key)

            # print("ffff")
            # # print(int(key)-1)
            # print(track_idx2)
            # print(det_idx)
            if track_idx < track_idx2:
                track_idx2 = track_idx


            list_t.append(track_idx)


            self.trackers[track_idx].update(detection=detections[det_idx])
            self.detections_matched_ids[det_idx] = self.trackers[track_idx].id

        
        


        print(matches)

        

        # not assigned detections: create new trackers POF
        assigned_det_idxs = set(matches[:, 1]) if len(matches) > 0 else []
        

        for det_idx in set(range(len(detections))).difference(assigned_det_idxs):
            det = detections[det_idx]
#            logger.info(det.score)
            # tracker = self.tracker_clss(box0=det.box,
            #                             score0=det.score,
            #                             class_id0=det.class_id,
            #                             **self.tracker_kwargs)
            
            # if det.class_id == "stuffed toy" and self.frame != 0:
            #     if not self.t_id in list_t and self.frame != 0:
            #         self.detections_matched_ids[det_idx] = self.t_id
            #         self.trackers[int(self.t_id)].update(detection=detections[det_idx])

            # else :
            tracker = self.tracker_clss(box0=det.box,
                        score0=det.score,
                        class_id0=det.class_id,
                        **self.tracker_kwargs)
            self.detections_matched_ids[det_idx] = tracker.id
            self.trackers.append(tracker)
        self.frame +=1 

#        logger.info(self.trackers)
        # unassigned trackers
        assigned_track_idxs = set(matches[:, 0]) if len(matches) > 0 else []
        for track_idx in set(range(len(self.trackers))).difference(assigned_track_idxs):
            self.trackers[track_idx].stale()

        # cleanup dead trackers
        self.cleanup_trackers()

        # log step timing
        elapsed = (time.time() - t0) * 1000.
        logger.debug(f'tracking step time: {elapsed:.3f} ms')

        if self.active_tracks(**self.active_tracks_kwargs) == [] :
            print("tracker_select")
            return self.trackers

        return self.active_tracks(**self.active_tracks_kwargs)
