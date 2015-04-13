#!/usr/bin/env python

"""Platform identification script

For use by a Makefile to enable differentiation

Author:  http://github.com/kbrafford
"""

import sys, platform

bits = platform.architecture()[0]
platform_id = sys.platform

print platform_id
print bits
