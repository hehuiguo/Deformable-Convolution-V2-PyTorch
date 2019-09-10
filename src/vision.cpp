
#include "deform_conv3d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv3d_forward", &deform_conv3d_forward, "deform_conv3d_forward");
  m.def("deform_conv3d_backward", &deform_conv3d_backward, "deform_conv3d_backward");
}
