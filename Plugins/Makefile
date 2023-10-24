
MANTA_DIR = /home/ian/projects/MaNTA
SUNDIALS_DIR = /home/ian/projects/sundials/install/include
TOML_DIR = /home/ian/projects/toml11/
EIGEN_DIR = /usr/include/eigen3

CXXFLAGS += -shared -fPIC -I$(MANTA_DIR) -I$(SUNDIALS_DIR) -I$(TOML_DIR) -I$(EIGEN_DIR)

%.so: %.cpp MantaPhysicsPlugin.hpp
	$(CXX) $(CXXFLAGS) -o $@ $^;
