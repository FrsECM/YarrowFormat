import os
from copy import deepcopy
from datetime import datetime
import pytest
from pydantic import ValidationError

from yarrow import *

@pytest.fixture
def contributor_base():
    return Contributor(
        name="Jean Claude Vandamme",
        human=True,
        email="jcv@vive-la-belgique.com")

@pytest.fixture
def yarrow_base(contributor_base):
    yarrow = YarrowDataset(
        info=Info(date_created=datetime.now(),source='Bruxelles'),
        images=[
            Image(file_name='image1.jpg',height=100,width=100,date_captured=datetime.now()),
            Image(file_name='image2.jpg',height=100,width=100,date_captured=datetime.now())],
        contributors=[contributor_base]
    )
    return yarrow

def test_create_multiple_annotations(yarrow_base:YarrowDataset,contributor_base:Contributor):
    """In this test we make sure adding annotations with different polygon shape is working.

    Args:
        yarrow (YarrowDataset): _description_
        contributor (Contributor): _description_
    """
    ann1 = Annotation(
        contributor_base,
        name='chaussure',
        images=yarrow_base.images,
        polygon=np.zeros((23,2))
    )
    ann2 = Annotation(
        contributor_base,
        name='chaussure',
        images=yarrow_base.images,
        polygon=np.zeros((10,2))
    )
    list_annotations = [ann1,ann2]
    yarrow_base.add_annotations(list_annotations)

