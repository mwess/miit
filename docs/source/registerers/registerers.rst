============
Registration
============

A key component of MIIT is the registration of serial sections to align the image spaces of spatial-omcis 
data.

-----------
GreedyFHist
-----------

GreedyFHist is an intensity based nonrigid registration algorithm that was developed together with MIIT. It supports
rigid, affine, and nonrigid registration. GreedyFHist has greedy as an external dependency.

--------------------------
OpenCVTutorialRegistration
--------------------------

This registration algorithm is based on OpenCV's tutorial for affine registration. It uses SURF features, RANSAC and flann.
It is not very accurate for registration of serial sections with tissue damage. However, for registration of differently cropped
images it works considerably well. 

--------
NiftyReg
--------

NiftyReg is a registration algorithm designed by ... for medical image registration. We implemented a wrapper method that calls NiftyReg. 
Therefore, NiftyReg is an external dependency. In our integration pipeline we use NiftyReg for registration of histology and msi data. (see example).
At the moment only rigid registration is supported.

----------------------------------
Manual landmark-based registration
----------------------------------

In case no accurate registration between images can be determined, we implemented an affine registration algorithm that uses manually
placed landmarks to register images. These landmarks can be determined using external tools such as Fiji.


------------------------------------------
Extending other to registration algorithms
------------------------------------------

Additional registration algorithms can be implemented as well. We provide the following basic structure in miit/registerers/base_registerer.py

.. code-block:: python

    @dataclass
    class Registerer(abc.ABC):
        """
        Provides methods that each registerer should implement. 
        """

        name: ClassVar[str]

        @abc.abstractmethod
        def register_images(self, 
                            moving_img: numpy.array, 
                            fixed_img: numpy.array, 
                            **kwargs: Dict)-> 'RegistrationResult':
            pass

        @abc.abstractmethod
        def transform_pointset(self, 
                               pointset: numpy.array, 
                               transformation: 'RegistrationResult', 
                               **kwargs: Dict) -> numpy.array:
            pass

        @abc.abstractmethod
        def transform_image(self, 
                            image: numpy.array, 
                            transformation: 'RegistrationResult', 
                            interpolation_mode: str, **kwargs: Dict) -> numpy.array:
            pass

        @classmethod
        @abc.abstractmethod
        def load_from_config(cls, config: Dict[str, Any]) -> 'Registerer':
            pass


For examples, see miit/registerers/. 

----------------------
Groupwise Registration
----------------------


Groupwise registration is not part of the base design of registration algorithms yet, though we provide a helper function in `miit/spatial_data/sections.py`.