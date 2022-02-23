import cv2
import time
import numpy as np
from computations import lsa_solve_lapjv, iou


class single_detection:
    def __init__(self, id, coord, l, h, conf):
        self.id = id
        self.coord = coord
        self.l = l
        self.h = h
        self.hist = None
        self.inactive_counter = 0
        self.is_active = False
        self.conf = conf
        self.is_confirmed = False

    def update(self, coord, hist, l, h, conf):
        #self.id = id
        self.hist = hist
        self.coord = coord
        self.l = l
        self.h = h
        self.conf = conf
        self.inactive_counter = 0
        self.is_active = True
        self.is_confirmed = True

    def __str__(self):
        return f"id: {self.id}, coord: {self.coord}, len: {self.l}, height: {self.h}"


class Tracker:

    def __init__(self, similarity_treshold, high_con_th, low_con_th, lifetime, alpha):
        self.similarity_treshold = similarity_treshold
        self.high_t = high_con_th
        self.low_t = low_con_th
        self.lifetime = lifetime
        self.alpha = float(alpha)
        self.beta = 1 - alpha
        self.active_tracks = []
        self.inactive_tracks = []
        self.last_id = 0

    def set_tracklets(self, tracks):
        self.active_tracks = tracks

    def update_det(self, detections):
        tmp_id = self.last_id
        for det in detections:
            det.id = tmp_id
            tmp_id += 1
        return detections

    def add_track(self, track):
        self.active_tracks.append(track)

    def get_tracks(self):
        return [track for track in self.active_tracks if track.is_active]

    def calculate_IOU(self, tracks, detections):
        start_time = time.time()
        iou_matrix = []
        for track in tracks:
            iou_vec = []
            for det in detections:
                IOU_distance = 1 - iou(track, det)
                iou_vec.append(IOU_distance)
            iou_matrix.append(iou_vec)
        print("--- %s seconds --iou-" % (time.time() - start_time))
        return iou_matrix

    def calcutale_Bhattacharyya_distance(self, tracks, detections):
        start_time = time.time()
        tracks_hist = []
        det_hist = []
        hist_sim = []
        for i in range(len(tracks)):
            hist_sim.append([1] * len(detections))
        for i in range(len(tracks)):
            for j in range(len(detections)):
                hist_sim[i][j] = cv2.compareHist(tracks[i].hist, detections[j].hist,
                                                 cv2.HISTCMP_BHATTACHARYYA)
        print("--- %s seconds --bh-" % (time.time() - start_time))
        return hist_sim

    def compute_cost_matrix(self, tracks, detections):
        return self.alpha * np.array(self.calculate_IOU(tracks, detections)) + self.beta * np.array(
            self.calcutale_Bhattacharyya_distance(tracks, detections))

    def Track(self, detections):
        if len(self.active_tracks) == 0:  # tracker initilization
            for det in detections:
                det.id = self.last_id
                det.is_active = True
                det.is_confirmed = True
                self.last_id += 1
                self.active_tracks.append(det)
        else:
            start_time = time.time()

            high_conf_det = []
            low_conf_det = []

            confirmed_tracks = []
            unconfirmed_tracks = []

            lost_tracks = []

            # division of tracks
            for track in self.active_tracks:
                if track.is_confirmed:
                    confirmed_tracks.append(track)
                else:
                    unconfirmed_tracks.append(track)

            # division of detections
            for det in detections:
                if det.conf > self.high_t:
                    high_conf_det.append(det)
                elif det.conf > self.low_t:
                    low_conf_det.append(det)

            unmatched_tracks = {i for i in range(len(confirmed_tracks))}
            unmatched_detections = {i for i in range(len(high_conf_det))}

            tracks_matched = set()
            detections_matched = set()

            # first matching with high conf
            if len(confirmed_tracks) > 0 and len(high_conf_det) > 0:
                matrix = self.compute_cost_matrix(confirmed_tracks, high_conf_det)
                print("--- %s seconds --cost matrix-" % (time.time() - start_time))
                start_time = time.time()
                r, c = lsa_solve_lapjv(matrix)
                indexes = [(row, col) for row, col in zip(r, c)]
                print("--- %s seconds --lap-" % (time.time() - start_time))
                for i in range(len(indexes)):
                    if matrix[indexes[i][0]][indexes[i][1]] < self.similarity_treshold:
                        cur_track, cur_det = confirmed_tracks[indexes[i][0]], high_conf_det[indexes[i][1]]
                        cur_track.update(cur_det.coord, cur_det.hist, cur_det.l, cur_det.h, cur_det.conf)
                        tracks_matched.add(indexes[i][0])
                        detections_matched.add(indexes[i][1])
                unmatched_detections = {x for x in range(len(high_conf_det))} - detections_matched
                unmatched_tracks = {x for x in range(len(confirmed_tracks))} - tracks_matched
                lost_tracks = [confirmed_tracks[i] for i in unmatched_tracks]

            # second matching with low conf detections
            if len(unmatched_tracks) > 0 and len(low_conf_det) > 0:
                remain_tracks = [confirmed_tracks[i] for i in unmatched_tracks]
                tracks_matched = set()
                matrix = self.calculate_IOU(remain_tracks, low_conf_det)
                r, c = lsa_solve_lapjv(np.array(matrix))
                indexes = [(row, col) for row, col in zip(r, c)]
                for i in range(len(indexes)):
                    if matrix[indexes[i][0]][indexes[i][1]] < self.similarity_treshold:
                        cur_track, cur_det = remain_tracks[indexes[i][0]], low_conf_det[indexes[i][1]]
                        cur_track.update(cur_det.coord, cur_det.hist, cur_det.l, cur_det.h, cur_det.conf)
                        tracks_matched.add(indexes[i][0])
                unmatched_tracks = {i for i in range(len(remain_tracks))} - tracks_matched
                lost_tracks = [remain_tracks[i] for i in unmatched_tracks]

            # matching for unconfirmed tracks
            if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
                remain_detections = [high_conf_det[i] for i in unmatched_detections]
                detections_matched = set()
                matrix = self.calculate_IOU(unconfirmed_tracks, remain_detections)
                r, c = lsa_solve_lapjv(np.array(matrix))
                indexes = [(row, col) for row, col in zip(r, c)]
                for i in range(len(indexes)):
                    if matrix[indexes[i][0]][indexes[i][1]] < self.similarity_treshold:
                        cur_track, cur_det = unconfirmed_tracks[indexes[i][0]], remain_detections[indexes[i][1]]
                        cur_track.update(cur_det.coord, cur_det.hist, cur_det.l, cur_det.h, cur_det.conf)
                        detections_matched.add(indexes[i][1])
                unmatched_detections = {x for x in range(len(remain_detections))} - detections_matched

            for track in lost_tracks:  # setting inactive tracks
                track.is_active = False

            remain_detections = [detections[i] for i in unmatched_detections]
            for det in remain_detections: # new detections
                det.id = self.last_id
                det.is_active = True
                self.last_id += 1
                self.active_tracks.append(det)

            #for i in unmatched_tracks:  # setting inactive tracks
            #    self.active_tracks[i].is_active = False
            #
            # for i in unmatched_detections:  # new detections
            #     detections[i].id = self.last_id
            #     detections[i].is_active = True
            #     self.last_id += 1
            #     self.active_tracks.append(detections[i])

            for track in self.active_tracks:  # updating old tracks' age
                if not track.is_active:
                    track.inactive_counter += 1
                    if track.inactive_counter >= self.lifetime: self.active_tracks.remove(track)
