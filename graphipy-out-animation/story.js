// Narrative for the intro animation and plain-language notes for the explorer.
// Edit texts here; app.js only reads this file.
//
// CLAIM BOUNDARY: the wording below deliberately separates what runs live, what is
// replayed from an audited archive, and what is designed but not implemented. Keep
// that separation if you edit these texts - it mirrors the project documentation.
//
// Community ids below match graphify-out/graph.json as rebuilt on 2026-10-20
// (1,486 nodes / 3,216 edges / 89 communities). If you re-run /graphify and the
// community count changes, re-check every id in communityColors and communityNotes.
window.STORY = (function () {
  'use strict';

  // The nine blocks of the tour. `anchor` is the real graph node the block flies to
  // when the intro ends, so the story stays attached to the actual software map.
  // Every anchor below was verified to exist in graphify-out/graph.json.
  var blocks = [
    { key: 'registry', name: 'SKU Registry', color: '#9d7bff',
      anchor: 'inventory_inventoryservice_register_product',
      short: 'The owner registers each product with reference photos' },
    { key: 'capture', name: 'Capture Checks', color: '#a6ff4d',
      anchor: 'experiments_opencv5_guided_scan_capture_gate_quality',
      short: 'OpenCV 5 measures focus, light and size' },
    { key: 'cloud', name: 'AWS Capture Gate', color: '#3ddc97',
      anchor: 'experiments_opencv5_guided_scan_aws_pilot_handler_lambda_handler',
      short: 'The same checks run deployed on AWS Lambda' },
    { key: 'detect', name: 'Find The Products', color: '#ff5fd2',
      anchor: 'vision_detect_products',
      short: 'One outline per physical item on the shelf' },
    { key: 'match', name: 'Suggest A SKU', color: '#4de3ff',
      anchor: 'sam_vision_samrecognizer_recognize',
      short: 'Compare each item against the registered photos' },
    { key: 'verify', name: 'Read The Label', color: '#8b7dff',
      anchor: 'local_features_labelverifier',
      short: 'A second opinion when two packages look alike' },
    { key: 'replay', name: 'Archived Evidence', color: '#ffc857',
      anchor: 'experiments_opencv5_guided_scan_replay',
      short: 'Saved model results, locked to the exact photo' },
    { key: 'approve', name: 'Human Approval', color: '#ff8c5a',
      anchor: 'db_create_scan_session',
      short: 'Nothing is counted until a person confirms it' },
    { key: 'ledger', name: 'Stock Movement', color: '#59c8ff',
      anchor: 'experiments_new_product_scanning_20261020_inventory_movement_design_20261020_add_subtract_ledger',
      short: 'Designed, not yet implemented' }
  ];

  // Ring layout for the intro stage: left, over the top, down to bottom-left.
  var R = 330;
  blocks.forEach(function (b, i) {
    var a = (180 - i * (360 / blocks.length)) * Math.PI / 180;
    b.pos = [Math.cos(a) * R, Math.sin(a) * R * 0.74, (i % 2 ? 40 : -40)];
  });

  var steps = [
    { kind: 'title', duration: 5200,
      kicker: 'A one-minute tour',
      title: 'Web&Shop · InventoryLens',
      text: 'A shop photographs its shelves, software proposes what it sees, and a person decides before any stock number changes. ' +
            'The cloud behind this text is the real software: every dot is a piece of code or documentation, every line a connection.' },

    { kind: 'block', block: 'registry', duration: 4400,
      kicker: 'Step 1 · Teaching it the products',
      title: 'SKU Registry',
      text: 'The shop owner registers each sellable product once, with up to four clear photographs from different angles. ' +
            'Nothing can be recognised that was never registered.' },

    { kind: 'block', block: 'capture', from: ['registry'], duration: 4400,
      kicker: 'Step 2 · Is the photo good enough?',
      title: 'Capture Checks',
      text: 'Before anything is analysed, OpenCV 5 decodes the photograph and measures its size, its sharpness and how dark or bright it is. ' +
            'It answers one question only: proceed, or retake.' },

    { kind: 'block', block: 'cloud', from: ['capture'], duration: 4600,
      kicker: 'Step 3 · The same code, in the cloud',
      title: 'AWS Capture Gate',
      text: 'Uploading a photo to a private S3 bucket triggers a Lambda function running that identical OpenCV 5 module in eu-north-1. ' +
            'On 176 photographs the cloud and the laptop agreed on every decision. This part is deployed and live.' },

    { kind: 'block', block: 'detect', duration: 4600,
      kicker: 'Step 4 · One box per real item',
      title: 'Find The Products',
      text: 'The hard part is not naming a product, it is separating them: two cans touching, one box behind another, a stack seen from the side. ' +
            'On the thirty reviewed scenes the detector proposed ninety-three boxes, and it does still miss items.' },

    { kind: 'block', block: 'match', from: ['detect', 'registry'], duration: 4600,
      kicker: 'Step 5 · What is this one?',
      title: 'Suggest A SKU',
      text: 'Each detected item is compared against every registered reference photograph, and the closest match is proposed with alternatives, ' +
            'ranked three ways: against plain reference crops, masked ones, and the two combined.' },

    { kind: 'block', block: 'verify', from: ['match'], duration: 4400,
      kicker: 'Step 6 · When two packages look alike',
      title: 'Read The Label',
      text: 'If the best two candidates are too close to call, OpenCV compares the printed detail on the label itself, ' +
            'matching keypoints geometrically. Exactly one SKU must verify, or the item stays unassigned.' },

    { kind: 'block', block: 'replay', duration: 4600,
      kicker: 'Step 7 · Honest about the evidence',
      title: 'Archived Evidence',
      text: 'Saved model results are shown only when a photograph’s SHA-256 matches an audited original exactly. ' +
            'They are replayed outputs, never fresh inference, and the interface says so on screen.' },

    { kind: 'block', block: 'approve', from: ['match', 'verify'], duration: 4600,
      kicker: 'Step 8 · A person decides',
      title: 'Human Approval',
      text: 'An employee splits merged boxes, adds missed items, corrects the SKU and the quantity, then approves. ' +
            'No proposal ever writes stock on its own, and re-approving the same scan cannot double-count it.' },

    { kind: 'block', block: 'ledger', from: ['approve'], duration: 4600,
      kicker: 'Step 9 · Designed, not built',
      title: 'Stock Movement',
      text: 'An approved count should become an ADD or SUBTRACT movement with an actor, a reason and a reversal path. ' +
            'A photograph cannot know the direction. This ledger is specified in detail and deliberately not yet implemented.' },

    { kind: 'outro', duration: 5200,
      kicker: 'In one sentence',
      title: 'Software proposes. People decide.',
      text: 'Behind these nine blocks are 1,486 pieces of code and documentation and 3,216 connections. ' +
            'Explore them below: live evidence, archived evidence and planned work are kept apart on purpose.' }
  ];

  // Communities that host the tour blocks take the block colours.
  var communityColors = {
    5: '#9d7bff', 50: '#a6ff4d', 49: '#3ddc97', 11: '#ff5fd2',
    16: '#4de3ff', 0: '#8b7dff', 39: '#ffc857', 8: '#ff8c5a', 54: '#59c8ff'
  };

  // Plain-language notes for the blocks of the map, one per community.
  var communityNotes = {
    0: 'The second opinion. When two packages look nearly identical, this compares the printed detail on the labels geometrically and refuses to answer unless exactly one product verifies.',
    1: 'Tooling that turns raw shelf photographs into a reviewable segmentation dataset: proposing outlines, staging them for the CVAT annotation tool, and checking the polygons are valid.',
    2: 'The renderer behind this very page: the 3D scene, the selection logic, the panel and the console helpers.',
    3: 'The separate review application for new products: it shows each scene, lets a person approve or correct a SKU identity, and records corrections apart from the original data.',
    4: 'Experiments comparing ways of cutting a product out of a photograph, scored against masks a human had already reviewed.',
    5: 'The service layer of the shop application. Every user action passes through here, which is also where it is checked that they are allowed to take it.',
    6: 'The hybrid pipeline: a compact detector paired with an image encoder, kept separate from the main application so it can be measured without disturbing it.',
    7: 'The October 17 diagnostic benchmark: a five-photograph comparison of three recognition approaches, scored side by side.',
    8: 'The database and sign-in core. All stored data passes through here, including the single transaction that records an approved count.',
    9: 'Shared helpers that read and convert detector datasets between formats without altering the source photographs.',
    10: 'Experiments on how tightly to crop a product before comparing it, and which encoder to use.',
    11: 'The detection pipeline of the live application: find the products, crop them, reject background, and build the board of reference views.',
    12: 'The October 17 benchmark again, prepared to run on a rented GPU instead of a laptop.',
    13: 'Head-to-head comparison of image encoders on the same photographs, including when the system should abstain rather than answer.',
    14: 'The Streamlit application shell: the owner and employee screens, image loading, and the pages everything else hangs from.',
    15: 'Training a segmentation model on the reviewed polygons, with checks that the dataset is intact first.',
    16: 'The recognition engine: it cuts each product out of the scene, turns the crop into numbers, and compares those numbers with every registered reference photograph.',
    17: 'Drawing the results: turning outlines into the annotated picture a person reviews, at the original image size.',
    18: 'The design of the add and subtract stock ledger, and the rule that a photograph alone never has a movement direction.',
    19: 'The deployed OpenCV 5 capture pilot: a private bucket, an event filter that cannot retrigger itself, and the 176-image audit.',
    20: 'The plan for evaluating thirty reviewed scenes on a GPU, and the readiness audit that had to pass first.',
    21: 'The competition demo storyboard and the replay contract that says which saved predictions may be shown and when.',
    22: 'Loading the detector: which weights, which fallback, and the checks that the right file was loaded.',
    23: 'The inference runtime for a rented GPU machine, including how several matchers are combined into one suggestion.',
    24: 'The new-product scanning experiment: its dataset audit, its scene evaluator, and the archives it produces.',
    25: 'Packaging work for GPU inference: bundling models and inputs, and refusing to proceed unless the GPU runtime genuinely works.',
    26: 'How the guided scan is put together: the shared capture module, the S3-to-Lambda flow, and the offline replay used to review it.',
    27: 'Scripts that assemble benchmark inputs and previews from existing photographs without modifying them.',
    28: 'The benchmark runner that drives the hybrid pipeline across a set of photographs.',
    29: 'A small demonstration page for segmentation, including adding a box by hand.',
    30: 'Offline tests for the cloud function, using a fake S3 so the whole decision path can be checked without touching AWS.',
    31: 'Tests of the approval flow itself: permissions, manual corrections, and the rule that changing the references invalidates a pending result.',
    32: 'The competition submission itself: the narrative, the technical report, and the architecture claim boundary.',
    33: 'The photography protocol and the matching thresholds: which four views to take, and how close a match must be before it counts.',
    34: 'The live guided-scan page: inspect a photograph, run the cloud capture check, and show saved proposals only for an exact hash match.',
    35: 'The headless browser test for this map, which drives the tour and the explorer and screenshots every step.',
    36: 'Evaluation of the NVIDIA Retail DINO detector on a small holdout, with the overlays a person reviewed.',
    37: 'The audit trail of the GPU benchmark: what was measured, what was recovered after a failure, and which claims the evidence does not support.',
    38: 'The review discipline for annotations: how photographs are grouped so related ones never straddle a split, and why every polygon is reviewed.',
    39: 'Assembling the archived prediction records that the review interface replays, keyed to exact image hashes.',
    40: 'The reference-retrieval pilot: how many reference photographs per product are enough, measured by leaving one out.',
    41: 'The isolated runner that performs scene detection and gallery ranking on a GPU and writes verifiable results.',
    42: 'Building the upload manifest and the deliberate retake controls, so every file sent to the cloud carries a hash that can be checked on the way back.',
    43: 'The YOLO detection experiment: one generic product class, evaluated with no scene crossing between training and test splits.',
    44: 'A feasibility check of what OpenCV 5 alone can recognise, including barcode reading and keypoint matching.',
    45: 'The limits the work operates under: the readiness audit, the spending ceiling, and the estimated cost of a GPU run.',
    46: 'Training the detector, with the dataset manifest validated before a single step runs.',
    47: 'The script that turns the knowledge graph into the payload this page loads, including the 3D layout.',
    48: 'Preparing annotation corrections for review, keeping the originals untouched.',
    49: 'The deployed cloud function. It validates the event, re-derives its own output path rather than trusting the caller, caps the read size, and returns a decision.',
    50: 'The capture quality module. Forty lines of OpenCV 5 that decide proceed or retake, and the only part of the vision stack that runs in the cloud today.',
    51: 'Scoring segmentation quality by matching predicted outlines to reviewed ones.',
    52: 'The end-to-end detector-plus-recogniser runner, with hashes recorded at every step.',
    53: 'The ideas behind the hybrid pipeline: which detector, which encoder, and how the stages are orchestrated.',
    54: 'What OpenCV 5 can contribute to recognition, and the append-only add/subtract ledger that an approved count should eventually produce.',
    55: 'The one-time cloud scene inference runner, its required paths, and the single-image preflight that must pass before a full run.',
    56: 'Reconciling a counted scene against the recorded inventory and reporting the difference.',
    57: 'The record of what was decided for the prototype and why, including which model comparisons are not yet conclusive.',
    58: 'The concepts behind the October benchmark: the three arms compared and the oracle arm that marks the ceiling.',
    59: 'The scope of the local prototype: that an approved count replaces a location total rather than adding to it.',
    60: 'The rules that stop double counting: two photographs of one arrangement are reconciled into a single total, never summed.',
    61: 'Building the offline HTML review of the latest recognition results.',
    62: 'Assembling the GPU inference bundle, hashing every file that goes into it.',
    63: 'Tests proving the separate matcher arms stay independent and are not allowed to contaminate each other.',
    64: 'The provisional architecture decisions: which recognition modes exist, and why rectangles were chosen over polygons for now.',
    65: 'Verifying a downloaded result archive by hash before any of it is imported.',
    66: 'Building the upload manifest for the capture pilot.',
    67: 'Building the segmentation training bundle from reviewed polygons only.',
    68: 'Materialising the grouped experiment dataset without modifying the source photographs.',
    69: 'The three-part photo handoff: 193 audited source paths covering 176 distinct images, with manifests and approved masks.',
    70: 'The security review workflow added to this repository: what it looks for, and how it filters false positives.',
    71: 'The AWS limits this project operates under: one selected region, service gaps, and what was and was not authorised.',
    72: 'Preparing the detector model for a GPU runtime, recording that the candidate file differs from the one previously proven to work.',
    73: 'The client that uploads exact bytes to the cloud on an explicit click, and rejects any answer whose hash disagrees.',
    74: 'The data contract behind the offline review page, including the rule that overlapping boxes can be excluded from a count.',
    75: 'This map itself: the page, the renderer, the dataset and the guided tour you just watched.',
    76: 'Packaging uploads for an alternative GPU provider without replacing existing files.',
    77: 'A small hash-verified bundle for re-running the five-photograph diagnostic remotely.',
    78: 'The shipped recognition defaults, and the note that a dependency floor of 4.7 does not prove OpenCV 5 is what runs.',
    79: 'The environment variables a GPU run needs before anything else will work.',
    80: 'A proposal-only dataset split, grouped so related photographs never straddle the boundary. It copies nothing.',
    81: 'Auditing downloaded cloud results against the local baseline, image by image.',
    82: 'An offline view of the duplicate annotation boxes on one photograph, shown unchanged for a person to judge.',
    83: 'Packaging the cloud function from pinned Linux wheels so the deployed code is reproducible.',
    84: 'Collecting outputs and weights into one archive before a rented machine is destroyed.',
    85: 'A gate that fails closed unless the GPU can actually execute a real operation.',
    86: 'Rendering the shared edge between two reviewed outlines, so a person can judge an overlap without the data being altered.',
    87: 'Choosing a GPU to rent and estimating what an hour of it costs, storage included.',
    88: 'The guided intro tour text for this page.'
  };

  // How to read each kind of connection, in plain words. {a} and {b} are the two ends.
  var relationNotes = {
    calls: '{a} runs {b}. When the first one works, it asks the second one to do part of the job.',
    contains: '{a} contains {b}: the second one is defined inside the first.',
    references: '{a} refers to {b}, for example as a marker, a type or a name it relies on.',
    imports: '{a} imports {b}: it pulls in that file so it can use what is inside.',
    imports_from: '{a} imports something from {b}.',
    rationale_for: '{a} is a note explaining why {b} exists or works the way it does.',
    conceptually_related_to: '{a} and {b} are related ideas, according to the documentation.',
    shares_data_with: '{a} and {b} exchange or share the same data.',
    indirect_call: '{a} ends up running {b} through another step in between.',
    method: '{b} is an operation that belongs to {a}.',
    uses: '{a} uses {b}.',
    cites: '{a} cites {b} as its source.',
    semantically_similar_to: '{a} and {b} solve nearly the same problem by different means. Found by comparing their meaning, not their wording.',
    inherits: '{a} is a specialised version of {b} and reuses its behaviour.',
    implements: '{a} implements {b}: it provides the concrete behaviour the second one describes.',
    defines: '{a} defines {b}.',
    participate_in: '{a} takes part in {b}.',
    form: '{a} is part of the group {b}.',
    depends_on: '{a} and {b} are connected through their contents.'
  };

  var kindNotes = {
    file: 'A source file: a container that holds related functions and classes.',
    function: 'A named piece of code that does one job. Other code can ask it to run.',
    class: 'A blueprint for a kind of object, bundling data and the operations on it.',
    symbol: 'A named value or marker used by the code in this file.',
    external: 'A building block from an outside library that this code uses.',
    concept: 'An idea, rule, measurement or decision taken from the project documentation.',
    rationale: 'A recorded reason: why something was done this way, or why an alternative was rejected.',
    document: 'A document that belongs to the project.',
    image: 'A picture or diagram from the documentation.'
  };

  var kindLabels = {
    file: 'File', function: 'Function', class: 'Class', symbol: 'Symbol', external: 'External symbol',
    concept: 'Concept', rationale: 'Rationale (why)', document: 'Document', image: 'Image'
  };

  var confidenceNotes = [
    'Extracted directly from the code or text. Certain.',
    'Inferred by the graph builder from the meaning of the text. Likely, not guaranteed.',
    'Ambiguous. The graph builder was not sure about this one.'
  ];

  return {
    blocks: blocks,
    steps: steps,
    communityColors: communityColors,
    communityNotes: communityNotes,
    relationNotes: relationNotes,
    kindNotes: kindNotes,
    kindLabels: kindLabels,
    confidenceNotes: confidenceNotes
  };
})();
