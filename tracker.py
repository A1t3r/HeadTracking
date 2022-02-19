import cv2
import time
import numpy as np
from computations import lsa_solve_lapjv, iou


class single_detection:
    def __init__(self, id, coord, l, h):
        self.id = id
        self.coord = coord
        self.l = l
        self.h = h
        self.hist = None
        self.inactive_counter = 0
        self.is_active = True

    def __str__(self):
        return f"id: {self.id}, coord: {self.coord}, len: {self.l}, height: {self.h}"


class Tracker:

    def __init__(self, similarity_treshold, lifetime, alpha):
        self.similarity_treshold = similarity_treshold
        self.lifetime = lifetime
        self.alpha = float(alpha)
        self.beta = 1 - alpha
        self.active_tracks = []
        self.inactive_tracks = []
        # self.m = Munkres()
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
                IOU_results_reversed = 1 - iou(track, det)
                iou_vec.append(IOU_results_reversed)
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
                self.last_id += 1
                self.active_tracks.append(det)
        else:
            start_time = time.time()
            matrix = self.compute_cost_matrix(self.active_tracks, detections)
            print("--- %s seconds --cost matrix-" % (time.time() - start_time))
            start_time = time.time()
            # indexes = self.m.compute(matrix.tolist())
            r, c = lsa_solve_lapjv(matrix)
            indexes = [(row, col) for row, col in zip(r, c)]
            print("--- %s seconds --lap-" % (time.time() - start_time))
            tracks_matched = set()
            detections_matched = set()
            for i in range(len(indexes)):
                if matrix[indexes[i][0]][indexes[i][1]] < self.similarity_treshold:
                    cur_track, cur_det = self.active_tracks[indexes[i][0]], detections[indexes[i][1]]
                    cur_track.coord = cur_det.coord
                    cur_track.hist = cur_det.hist
                    cur_track.l = cur_det.l
                    cur_track.h = cur_det.h
                    cur_track.is_active = True
                    cur_track.inactive_counter = 0
                    tracks_matched.add(indexes[i][0])
                    detections_matched.add(indexes[i][1])
            # print(tracks_matched)
            # print(detections_matched)

            unmatched_tracks = set([x for x in range(len(self.active_tracks))]) - tracks_matched
            unmatched_detections = set([x for x in range(len(detections))]) - detections_matched

            for i in unmatched_tracks:  # setting inactive tracks
                self.active_tracks[i].is_active = False

            for i in unmatched_detections:  # new detections
                detections[i].id = self.last_id
                detections[i].is_active = True
                self.last_id += 1
                self.active_tracks.append(detections[i])

            for track in self.active_tracks:  # updating old tracks' age
                if not track.is_active:
                    track.inactive_counter += 1
                    if track.inactive_counter >= self.lifetime: self.active_tracks.remove(track)
