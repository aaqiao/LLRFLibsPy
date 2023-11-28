# LLRF Algorithm Python Library (**LLRFLibsPy**)
The PDF version of the manual and an introduction can be found in the `doc` folder. The webpage-based documentation can be found at https://aaqiao.github.io/LLRFLibsPy/.

## Description of Files and Folders
- `src`:     folder containing source files of the library.
- `example`: folder containing examples for demonstrating the library.
- `doc`:     folder containing the documentation.
- `docs`:    folder containing the HTML manual.

## Contents of LLRFLibsPy
**LLRFLibsPy** consists of the following modules.
| Module        |Description                            |
|:--------------|:--------------------------------------|
| `rf_calib`    |RF calibrations like virtual probe, RF actuator offset/imbalance, forward and reflected, and power calibrations.|
| `rf_control`  |Design and analyze RF feedback/feedforward controllers.|
| `rf_det_act`  |Measure RF amplitude and phase from ADC samples.|
| `rf_fit`      |Fit data to sine/cosine, circle, ellipse or Gaussian functions.|
| `rf_misc`     |Save/read data to/from Matlab files.|
| `rf_noise`    |Analyze, generate and filter noise.|
| `rf_plot`     |Plotting functions for internal use.|
| `rf_sim`      |Simulate the RF cavity response in the presence of RF drive and beam loading.|
| `rf_sysid`    |Identify the RF system transfer function and characteristic parameters.|

## Disclaimer (see the **LICENSE** file)
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

