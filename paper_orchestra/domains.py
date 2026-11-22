"""Per-domain experiment recipes: which bundled scripts run, in what order, what each needs and writes, which figures the
results feed, and the domain's own wording for the planner and the writer.

A recipe is data.  The planner (outline_agent) and the writer read it; adding a domain means adding an entry here and the
scripts it names to examples/experiments/.  "microscopy" reproduces the leukemia chain exactly as it ran before domains
existed; "photos" is the chain for photographs of people or objects (a person/background audit and probe).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Modules the experiment scripts import; they are uploaded with them and never planned as experiments of their own.
HELPER_SCRIPTS = {
    "train_efficientvit.py", "stain_render.py", "int8_calibration.py", "kaggle_data.py", "zenodo_data.py", "data_source.py",
    "imgprep.py", "stats_utils.py", "cvlib.py", "person_mask.py", "photo_conditions.py", "pose_audit.py",
}


@dataclass
class Domain:
    name: str
    outputs: dict[str, list[str]]  # script -> the files it writes (the planner's guess is replaced by these)
    targets: dict[str, str]  # script -> colab | jetson (also the order the experiments run in)
    names: dict[str, str]  # chain script -> experiment name
    chain: tuple[str, ...]  # scripts added in front of the edge experiments when Colab trains the model
    needs: dict[str, tuple[str, ...]]  # script -> scripts whose outputs it needs, when chained
    edge: tuple[str, ...]  # a plan of one of these makes the chain necessary
    required: tuple[str, ...]  # the chain exists only if all of these scripts are available
    chained_targets: dict[str, str] = field(default_factory=dict)  # target overrides when chained
    figures: dict[str, tuple[str, str]] = field(default_factory=dict)  # figure_type -> (script, caption)
    planner_section: str = ""  # the domain's paragraph of the planner prompt (figures and chain)
    writer_notes: str = ""  # the domain's reporting rules for the writer
    preprocess: str = "squash"  # PREPROCESS for the experiments (imgprep.py)
    show_images: bool = True  # may image montages of the data appear in the paper
    env: dict[str, str] = field(default_factory=dict)  # extra env vars for the Colab scripts


MICROSCOPY = Domain(
    name="microscopy",
    outputs={
        "prepare_dataset.py": ["test_data.zip", "calib_data.zip", "split_info.json"],
        "dataset_stats.py": ["dataset_stats.json", "dataset_stats.png", "augmentation_examples.png"],
        "train_baseline.py": ["training_history_baseline.json", "model_baseline.pt"],
        "train_augmented.py": ["training_history.json", "model.pt", "model.onnx"],
        "triton_benchmark.py": ["benchmark_results.json"],
        "morphology_probe.py": ["probe_results.json", "probe_examples.png"],
    },
    targets={
        "prepare_dataset.py": "colab", "dataset_stats.py": "colab", "train_baseline.py": "colab",
        "train_augmented.py": "colab", "triton_benchmark.py": "jetson", "morphology_probe.py": "jetson",
    },
    names={
        "prepare_dataset.py": "Download the dataset from Kaggle and split it",
        "dataset_stats.py": "Statistics of the training backgrounds",
        "train_baseline.py": "Train EfficientViT-B0 without background augmentation (baseline)",
        "train_augmented.py": "Train EfficientViT-B0 with background/stain augmentation",
    },
    chain=("prepare_dataset.py", "dataset_stats.py", "train_baseline.py", "train_augmented.py"),
    needs={
        "dataset_stats.py": ("prepare_dataset.py",), "train_baseline.py": ("prepare_dataset.py",),
        "train_augmented.py": ("prepare_dataset.py",),
        "triton_benchmark.py": ("prepare_dataset.py", "train_augmented.py"),
        "morphology_probe.py": ("prepare_dataset.py", "train_augmented.py", "train_baseline.py"),
    },
    edge=("triton_benchmark.py", "morphology_probe.py"),
    required=("prepare_dataset.py", "train_augmented.py"),
    chained_targets={"morphology_probe.py": "colab"},
    figures={
        "dataset_stats": ("dataset_stats.py", "Background statistics of the training images per class (L*, a*, b*) and the accuracy of a linear classifier given only the background or only the cells."),
        "augmentation_examples": ("dataset_stats.py", "One image per class with its background (and cell colour) re-rendered on the Romanowsky protocols used for augmentation, and on held-out protocols."),
        "training_curves_baseline": ("train_baseline.py", "Training and validation loss and accuracy of the baseline EfficientViT-B0 per epoch, trained on Google Colab."),
        "training_curves": ("train_augmented.py", "Training and validation loss and accuracy of the background-augmented EfficientViT-B0 per epoch, trained on Google Colab."),
        "confusion_matrix": ("triton_benchmark.py", "Confusion matrices of the deployed variants on the held-out test images, from inference on the Jetson Orin Nano."),
        "roc_curve": ("triton_benchmark.py", "One-vs-rest ROC curves of the deployed variants on the held-out test images, from inference on the Jetson Orin Nano."),
        "morphology_probe": ("morphology_probe.py", "Accuracy of the baseline and the background-augmented model with the cells masked, kept on a neutral background, and re-rendered on stain protocols seen and not seen in training."),
        "morphology_examples": ("morphology_probe.py", "Example images for each probe condition: original, cells masked, cells only and re-rendered backgrounds."),
    },
    planner_section='''\
Result figures are drawn by code from the measurements, so name the experiment that
produces the data in the figure's "data_source" (e.g. "exp1") and use these figure_types:
  confusion_matrix   confusion matrix of every deployed variant      (needs triton_benchmark.py with BENCH_TEST_DIR)
  roc_curve          one-vs-rest ROC curves with AUC                  (same experiment)
  dataset_stats      background statistics per class and a background-only classifier (needs dataset_stats.py)
  augmentation_examples  one image per class re-rendered on several staining protocols (same experiment)
  morphology_probe   accuracy of the baseline and the background-augmented model with the cells masked, on a
                     neutral background and on stain protocols never used in training  (needs morphology_probe.py)
  morphology_examples  image montage of the original, masked and re-backgrounded cells (same experiment)
  training_curves    training/validation loss and accuracy per epoch of the augmented model (train_augmented.py)
  training_curves_baseline   the same for the baseline model (train_baseline.py)
When the matching script is in the Available scripts list, plan it:
  triton_benchmark.py  "jetson"  expected_outputs ["benchmark_results.json"]
If Colab training is available (see the requirements below), the experiments run in this chain:
  prepare_dataset.py   "colab"   downloads the dataset from Kaggle, splits it, packs the Jetson's test and calibration
                       images   expected_outputs ["test_data.zip", "calib_data.zip", "split_info.json"]
  dataset_stats.py     "colab"   statistics of the training backgrounds; runs BEFORE the training
  train_baseline.py    "colab"   trains without background augmentation   depends_on the dataset experiment
  train_augmented.py   "colab"   trains with background/stain augmentation depends_on the dataset experiment
  triton_benchmark.py  "jetson"  benchmarks the AUGMENTED model   depends_on the dataset and train_augmented.py experiments
  morphology_probe.py  "colab"   probes both trained models, with statistics   depends_on the dataset, baseline
                                 and augmented experiments
Without Colab training, morphology_probe.py is a "jetson" experiment on the model already there.
Plan the figures above for them (one of each), placed in the experiments/results sections.  The research question
about whether the model learns cell morphology or the stain/background is answered from dataset_stats.py (what the
backgrounds carry) and morphology_probe.py (how both models behave on held-out stains), never from anything else.
''',
    writer_notes='''
Background and morphology (the question of whether the model reads the cells or the stain):
- "dataset_stats" holds the statistics of the training backgrounds before any training: per-class
  background colour, Kruskal-Wallis tests, and a linear classifier given only the background or only
  the cells (cross-validated, with a permutation test).  Say what it measured and give its numbers;
  never claim more than the tests support.
- "morphology_probe" holds two models, "baseline" (no augmentation) and "augmented" (training images
  re-rendered on four stain protocols), both probed on the same test images.  Only the protocols listed
  under "held_out" were never used in training; the verdict about robustness rests on those, not on the
  four "train" protocols.  Quote each model's "verdict" and the "comparison" verdict, the confidence
  intervals and the McNemar p-values; do not describe a difference as real when its interval or p-value
  does not support it.
- The stain protocols are synthetic approximations of Romanowsky stains, not real stained slides; say so.
- The augmented model is the one deployed to the Jetson; the baseline was not benchmarked there.
''',
)

PHOTOS = Domain(
    name="photos",
    outputs={
        "prepare_data.py": ["test_data.zip", "calib_data.zip", "split_info.json"],
        "dataset_audit.py": ["dataset_audit.json", "dataset_audit.png"],
        "pose_benchmark.py": ["pose_benchmark.json", "pose_benchmark.png"],
        "masked_segmentation.py": ["masked_segmentation.json"],
        "train_cv.py": ["cv_results.json", "selected.json", "test_results.json", "training_history.json", "model.pt", "model.onnx"],
        "shortcut_probe.py": ["probe_results.json"],
        "triton_benchmark.py": ["benchmark_results.json"],
    },
    targets={
        "prepare_data.py": "colab", "dataset_audit.py": "colab", "pose_benchmark.py": "colab", "masked_segmentation.py": "colab", "train_cv.py": "colab",
        "shortcut_probe.py": "colab", "triton_benchmark.py": "jetson",
    },
    names={
        "prepare_data.py": "Download the dataset and split it",
        "dataset_audit.py": "Audit of the training images (background, person, metadata)",
        "pose_benchmark.py": "Benchmark of body-keypoint detectors (MediaPipe Pose, RTMPose, ViTPose, Keypoint R-CNN): speed, availability, agreement, leg geometry",
        "masked_segmentation.py": "Train segmentation networks on photographs with every person masked out: can they still tell the class?",
        "train_cv.py": "Compare backbones by session-aware cross-validation; fit and export the best",
        "shortcut_probe.py": "Probe baseline and background-randomised training out-of-fold",
    },
    chain=("prepare_data.py", "dataset_audit.py", "pose_benchmark.py", "masked_segmentation.py", "train_cv.py", "shortcut_probe.py"),
    needs={
        "dataset_audit.py": ("prepare_data.py",), "pose_benchmark.py": ("prepare_data.py",), "masked_segmentation.py": ("prepare_data.py",), "train_cv.py": ("prepare_data.py",),
        "shortcut_probe.py": ("prepare_data.py", "train_cv.py"),
        "triton_benchmark.py": ("prepare_data.py", "train_cv.py"),
    },
    edge=("triton_benchmark.py", "shortcut_probe.py"),
    required=("prepare_data.py", "train_cv.py"),
    figures={
        "dataset_audit": ("dataset_audit.py", "Per-class statistics of the training photographs (background lightness, share of the image covered by the person, file size, capture day) and the cross-validated accuracy of linear classifiers given only the background, only the person, only the metadata, or all of them."),
        "pose_benchmark": ("pose_benchmark.py", "Body-keypoint detectors compared on the training photographs: how often both knees are found per class, median time per image, and the cross-validated accuracy of a linear classifier on each detector's leg geometry against the majority baseline."),
        "cv_comparison": ("train_cv.py", "Backbones compared by repeated session-aware cross-validation on the training images (balanced accuracy and macro-F1 with 95% bootstrap intervals), with the majority-class and frozen-embedding references."),
        "training_curves": ("train_cv.py", "Training and validation loss and accuracy per epoch of the selected backbone, averaged over the folds and repeats of the cross-validation."),
        "shortcut_probe": ("shortcut_probe.py", "Out-of-fold accuracy of the baseline and the background-randomised recipe with the person masked, on other backgrounds, on noise, in grayscale and blurred; hatched bars are backgrounds never used in training."),
        "confusion_matrix": ("triton_benchmark.py", "Confusion matrices of the deployed variants on the held-out test images, from inference on the Jetson Orin Nano."),
        "roc_curve": ("triton_benchmark.py", "One-vs-rest ROC curves of the deployed variants on the held-out test images, from inference on the Jetson Orin Nano."),
    },
    planner_section='''\
Result figures are drawn by code from the measurements, so name the experiment that
produces the data in the figure's "data_source" (e.g. "exp1") and use these figure_types:
  confusion_matrix   confusion matrix of every deployed variant      (needs triton_benchmark.py with BENCH_TEST_DIR)
  roc_curve          one-vs-rest ROC curves with AUC                  (same experiment)
  dataset_audit      statistics of the background, the person and the file metadata, and what linear classifiers
                     learn from each (needs dataset_audit.py)
  pose_benchmark     keypoint detectors compared: detection rates, time per image and the classifier on each
                     detector's leg geometry (needs pose_benchmark.py)
  cv_comparison      the backbones compared by cross-validation, with the reference rows (needs train_cv.py)
  training_curves    cross-validation training/validation curves of the selected backbone (train_cv.py)
  shortcut_probe     out-of-fold accuracy of the baseline and the background-randomised recipe with the person
                     masked, on other backgrounds, on noise, grayscale and blurred (needs shortcut_probe.py)
When the matching script is in the Available scripts list, plan it:
  triton_benchmark.py  "jetson"  expected_outputs ["benchmark_results.json"]
If Colab training is available (see the requirements below), the experiments run in this chain:
  prepare_data.py      "colab"   downloads the dataset, splits it, packs the Jetson's test and calibration images
                       expected_outputs ["test_data.zip", "calib_data.zip", "split_info.json"]
  dataset_audit.py     "colab"   audit of the training images before any model is trained; depends_on the dataset experiment
  pose_benchmark.py    "colab"   benchmark of body-keypoint detectors (MediaPipe Pose, RTMPose, ViTPose and the Keypoint R-CNN
                       reference) on the training images; runs BEFORE the training; depends_on the dataset experiment
  train_cv.py          "colab"   backbones compared by session-aware cross-validation on the training images; fits and
                       exports the best and scores the test images once; depends_on the dataset experiment
  shortcut_probe.py    "colab"   baseline vs background-randomised training, probed out-of-fold; depends_on the dataset
                       and train_cv.py experiments
  triton_benchmark.py  "jetson"  benchmarks the selected model; depends_on the dataset and train_cv.py experiments
Plan the figures above for them (one of each), placed in the experiments/results sections.  Image montages are never
planned: the photographs show people.  The research question about whether the model reads the subject or the
surroundings is answered from dataset_audit.py and pose_benchmark.py (what the background, the metadata, the framing and the
leg geometry carry) and shortcut_probe.py (what the models rely on), never from anything else.
''',
    writer_notes='''
Photographs, shortcuts and small data:
- The data are photographs of people.  Never describe, reproduce or include a photograph.  Do not invent a consent, ethics
  approval or data-protection statement: write the line "[Ethics and consent statement to be provided by the authors.]".
- "split_info" says how the split was made and whether the official split shares capture sessions between train and test
  ("groups"); report that plainly, since it can inflate the test accuracy.
- "cv_results" holds the backbones compared by repeated session-aware cross-validation on the training images only, with
  95% bootstrap intervals, and two reference rows.  The backbone was chosen by the stated rule before the test images were
  looked at.  "test_results" is the single scoring of the final model on the test images: with so few images its interval is
  wide, so give the interval and do not rank models by it.
- "dataset_audit" holds statistics before any training: Kruskal-Wallis tests, and linear classifiers given only the
  background, only the person, or only the file metadata, each with a permutation test.  Report what each measured; a
  classifier at the majority baseline means that feature set carries no class information.
- "pose_benchmark" compares body-keypoint detectors (MediaPipe Pose, RTMPose-m and -x, ViTPose-B and ViTPose+-H, with Keypoint R-CNN
  as the reference) on the same training photographs.  There is NO keypoint ground truth, so it does not measure keypoint
  accuracy and you must not call any detector "more accurate".  It measures: time per image (person finding included; ViTPose
  uses the Keypoint R-CNN person box and its time includes that step; each entry names the device it ran on, and MediaPipe runs
  on the CPU, so say that its time is not comparable with the GPU ones); how often both knees, ankles and hips are found per
  class at each detector's own confidence threshold (the scores mean different things, so the thresholds differ: say so);
  whether finding the knees and ankles depends on the class (a framing cue); the agreement between detectors (median distance
  in pixels at a 512-pixel image height, and the rank correlation of the leg measures), which shows that they locate the same
  joints and not that the joints are right; and the leg geometry per class with a cross-validated linear classifier on the
  images in which all lower-leg keypoints were found, against the majority baseline of those images.  Quote the images per class
  that entered each classifier.  Keypoint errors of a few pixels are comparable to the angles that separate varus and valgus
  from normal, so nothing here is a clinical measurement, and the detectors have no heel or foot points.
- "probe_results" holds two recipes ("baseline", "augmented": background randomisation in training) evaluated out-of-fold
  under conditions of the photographs.  Only the conditions marked "held_out" use backgrounds never used in training.  The
  backgrounds are synthetic compositions, not real rooms; say so.  Quote each recipe's verdict, the comparison verdict, the
  intervals and the McNemar p-values; do not call a difference real when its interval or p-value does not support it.
- "agreement_with_colab_test_predictions", when present in the benchmark results, says how many Jetson predictions equal the
  predictions of the trained PyTorch model on the same test images; report disagreement if there is any.
- Reading the probe:
  * "person_masked" replaces the person by a flat colour, so the person's silhouette (size, position, shape) stays in the
    image; accuracy above the majority baseline there can come from the silhouette and is not evidence that the room carries
    class information (the audit says whether it does).  "scene_only", when present, removes the person by inpainting.
  * "person_box_crop" removes the person's scale and position in the frame but keeps the box's aspect ratio; never say it
    removes the framing.
  * The verdict "robust to the tested changes" means accuracy survives the removal or replacement of the background and colour; it
    does not show that the model reads the legs, because silhouette and aspect ratio remain.  Say exactly that.
  * Report the baseline's loss under a condition (for example on noise or in grayscale) as brittleness to that change, with the
    interval and p-value.
- A backbone at chance (balanced accuracy near the majority baseline and an AUC near 0.5) did not train under the shared
  recipe; say so.  Do not conclude that its architecture is unsuitable, and do not rank architectures whose intervals overlap.
- A linear classifier on a handful of audit features can match or beat fine-tuned networks; compare them only with the caveat
  that the two use different folds.
- The Jetson benchmarks the model selected by cross-validation; no other model was deployed.
''',
    preprocess="letterbox",
    show_images=False,
)

DOMAINS: dict[str, Domain] = {"microscopy": MICROSCOPY, "photos": PHOTOS}


def domain_of(exp) -> Domain:
    """The recipe for the experiment requirements (microscopy when the form names none)."""
    name = getattr(getattr(exp, "dataset", None), "domain", "") or "microscopy"
    return DOMAINS.get(name, MICROSCOPY)
