# File containing the core classes used in project, most notably SpatialPointVector.
import numpy as np
import warnings


class SpatialPointVector:
    # Class representing n points with spatial coordinates in dim-d space; Main datum (pointArray) is a 2-d
    # NumPy array with dimensions (n, dim) representing the j-th point and its spatial coordinates (0<=j<n).
    # TODO: Seems to be slow. Make more efficient.
    # (?) TODO: Make masked
    def __init__(self, data=None, n=None, dim=None, copy=False):
        if data is not None:
            if data.ndim != 1 and data.ndim != 2:
                raise TypeError("data does not have the correct shape.")

            elif data.ndim == 2 and (dim is not None):
                if data.shape[1] == dim:
                    array = data
                    self.dim = dim
                elif data.shape[1] != dim and data.shape[0] == dim:
                    array = data.T
                    self.dim = dim
                else:
                    array = data
                    self.dim = array.shape[1]
                    warnings.warn("The shape of data and dim do not fit each other.")

            elif data.ndim == 2 and dim is None:
                array = data
                self.dim = array.shape[1]

            elif (data.ndim == 1 and dim is None) or (data.ndim == 1 and dim == 1):
                array = data
                self.dim = 1

            else:
                array = data
                self.dim = 1
                warnings.warn("The shape of data and dim do not fit each other.")

        elif n is not None and dim is not None:
            array = np.zeros((n, dim))
            self.dim = dim

        else:
            raise TypeError("PointArray could not be created from passed arguments.")

        self.PointArray = np.array(array, copy=copy)
        self.binarized = True

        if self.dim >= 2:
            self.x = self.PointArray[:, 0]
            self.y = self.PointArray[:, 1]

        if self.dim >= 3:
            self.z = self.PointArray[:, 2]

    def __getitem__(self, item):
        return self.PointArray[item]

    def __setitem__(self, key, value):
        self.PointArray[key] = value

    def __str__(self):
        return self.PointArray.__str__()

    def __repr__(self):
        return self.PointArray.__repr__()

    def __len__(self):
        return self.PointArray.shape[0]

    """
    def Shave(self,cutoff):
        # Removes points with values which deviate absolutely too far from the dimension arithmetic mean as determined
        # by cutoff.
        # TODO: Add cutoff_types which allow more types of cutoff
        ma.masked_where(np.fabs( np.add( self.x.filled(0.), -self.x.mean() ) )>cutoff, self.x, copy=False)
        print(ma.getmask(x))
        ma.masked_where(ma.fabs(ma.add(self.y,-self.y.mean()))>cutoff, self.y, copy=False)
        ma.masked_where(ma.fabs(ma.add(self.z,-self.z.mean()))>cutoff, self.z, copy=False)

        ma.compress_rows(self.pointArray)
    """

    def get_axis(self, n, copy=False):
        return np.array(self.PointArray[:, n], copy=copy)
