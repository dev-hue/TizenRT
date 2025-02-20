/****************************************************************************
 *
 * Copyright 2019 Samsung Electronics All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied. See the License for the specific
 * language governing permissions and limitations under the License.
 *
 ****************************************************************************/
/****************************************************************************
 * arch/arm/src/armv8-m/up_vectors.c
 *
 *   Copyright (C) 2012 Michael Smith. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name NuttX nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/

/****************************************************************************
 * Included Files
 ****************************************************************************/

#include <tinyara/config.h>

#include "chip.h"

/************************************************************************************
 * Preprocessor Definitions
 ************************************************************************************/

#define IDLE_STACK      ((unsigned)&_ebss+CONFIG_IDLETHREAD_STACKSIZE-4)

#ifndef ARMV8M_PERIPHERAL_INTERRUPTS
#error ARMV8M_PERIPHERAL_INTERRUPTS must be defined to the number of I/O interrupts to be supported
#endif

/****************************************************************************
 * Public Functions
 ****************************************************************************/

/* Chip-specific entrypoint */

extern void __start(void);

/* Common exception entrypoint */

extern void exception_common(void);

/************************************************************************************
 * Public data
 ************************************************************************************/

/* Provided by the linker script to indicate the end of the BSS */

extern char _ebss;

/* The v7m vector table consists of an array of function pointers, with the first
 * slot (vector zero) used to hold the initial stack pointer.
 *
 * As all exceptions (interrupts) are routed via exception_common, we just need to
 * fill this array with pointers to it.
 *
 * Note that the [ ... ] designated initialiser is a GCC extension.
 */
#if 0
unsigned _vectors[] __attribute__((section(".vectors"))) = {
	/* Initial stack */

	0x10004FFC,
	/* Reset exception handler */
	(unsigned) &__start,
	/* Vectors 2 - n point directly at the generic handler */
	[2 ...(15 + ARMV8M_PERIPHERAL_INTERRUPTS)] = (unsigned) &exception_common
};
#endif
