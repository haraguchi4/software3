# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.26.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.26.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/haraguchiryodai/Downloads/ソフトウェア演習3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/haraguchiryodai/Downloads/ソフトウェア演習3/build

# Include any dependencies generated for this target.
include CMakeFiles/diff2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/diff2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/diff2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/diff2.dir/flags.make

CMakeFiles/diff2.dir/src/diff2.cpp.o: CMakeFiles/diff2.dir/flags.make
CMakeFiles/diff2.dir/src/diff2.cpp.o: /Users/haraguchiryodai/Downloads/ソフトウェア演習3/src/diff2.cpp
CMakeFiles/diff2.dir/src/diff2.cpp.o: CMakeFiles/diff2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/haraguchiryodai/Downloads/ソフトウェア演習3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/diff2.dir/src/diff2.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/diff2.dir/src/diff2.cpp.o -MF CMakeFiles/diff2.dir/src/diff2.cpp.o.d -o CMakeFiles/diff2.dir/src/diff2.cpp.o -c /Users/haraguchiryodai/Downloads/ソフトウェア演習3/src/diff2.cpp

CMakeFiles/diff2.dir/src/diff2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/diff2.dir/src/diff2.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/haraguchiryodai/Downloads/ソフトウェア演習3/src/diff2.cpp > CMakeFiles/diff2.dir/src/diff2.cpp.i

CMakeFiles/diff2.dir/src/diff2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/diff2.dir/src/diff2.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/haraguchiryodai/Downloads/ソフトウェア演習3/src/diff2.cpp -o CMakeFiles/diff2.dir/src/diff2.cpp.s

# Object files for target diff2
diff2_OBJECTS = \
"CMakeFiles/diff2.dir/src/diff2.cpp.o"

# External object files for target diff2
diff2_EXTERNAL_OBJECTS =

diff2: CMakeFiles/diff2.dir/src/diff2.cpp.o
diff2: CMakeFiles/diff2.dir/build.make
diff2: CMakeFiles/diff2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/haraguchiryodai/Downloads/ソフトウェア演習3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable diff2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/diff2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/diff2.dir/build: diff2
.PHONY : CMakeFiles/diff2.dir/build

CMakeFiles/diff2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/diff2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/diff2.dir/clean

CMakeFiles/diff2.dir/depend:
	cd /Users/haraguchiryodai/Downloads/ソフトウェア演習3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/haraguchiryodai/Downloads/ソフトウェア演習3 /Users/haraguchiryodai/Downloads/ソフトウェア演習3 /Users/haraguchiryodai/Downloads/ソフトウェア演習3/build /Users/haraguchiryodai/Downloads/ソフトウェア演習3/build /Users/haraguchiryodai/Downloads/ソフトウェア演習3/build/CMakeFiles/diff2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/diff2.dir/depend

