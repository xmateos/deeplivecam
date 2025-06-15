from insightface.app import FaceAnalysis
from insightface.app.common import Face

class FaceAnalysisUnlimited(FaceAnalysis):
    def get(self, img_bgr, max_num=0, **kwargs):
        bboxes, kpss = self.det_model.detect(img_bgr, max_num=max_num, metric='default')
        #use_max = max_num if max_num > 0 else 10
        #bboxes, kpss = self.det_model.detect(img_bgr, max_num=use_max, metric='default')
        #print("▶️ Detected ", bboxes.shape[0], " faces")
        if bboxes.shape[0] == 0:
            return []
        faces = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for task, model in self.models.items():
                if task == 'detection':
                    continue
                model.get(img_bgr, face)
            faces.append(face)
        return faces
