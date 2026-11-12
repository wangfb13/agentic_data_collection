"""A reviewed view must not hide later changes in original physical evidence."""
import json
from pathlib import Path
import pytest
from bottle_scan.skill_loop import source_snapshot
from bottle_scan.skill_label_review import apply_review
from tests import test_skill_label_review as review_fixtures


def test_frozen_view_snapshot_revalidates_original_source():
    fixture=review_fixtures.LabelReviewTests()
    fixture.setUp()
    try:
        original,review,_=fixture.prepared()
        answers=fixture.annotate(review)
        view=fixture.root/'view_snapshot'
        apply_review(review,answers,view)
        assert source_snapshot(view)['trials']
        path=next((original/'trials').glob('*/meta.json'))
        row=json.loads(path.read_text());row['status']='changed-after-review'
        path.write_text(json.dumps(row))
        with pytest.raises(ValueError,match='source trial metadata changed'):
            source_snapshot(view)
    finally:
        fixture.tearDown()
