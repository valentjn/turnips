#!/usr/bin/python3

import functools
import multiprocessing
import random
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate



# def runWorker(numberOfWorkers, workerId):
#   cmd = ["./turnips", "bruteForce", str(numberOfWorkers), str(workerId), "2", "9999",
#       "92", "92",   "82", "79",   "76", "118",   "0", "0",   "0", "0",   "0", "0",   "0", "0"]
#       # 148 271 145
#   process = subprocess.run(cmd, stdout=subprocess.PIPE)
#   a = process.stdout.decode()
#   print("{}: {}, #{}#, #{}#".format(workerId, len(a), a[:100], a[-100:]))
#   matches = np.reshape(np.fromstring(process.stdout.decode(), dtype=int, sep=" "), [-1, 17])
#   return matches



# def main():
#   """
# import numpy as np
# a = np.load("matches.npy")
# np.histogram(a[:,2], [-0.5, 0.5, 1.5, 2.5, 3.5])[0] / a.shape[0]
# np.amin(np.amax(a[a[:,2]==1,4:], axis=1))
# np.mean(np.amax(a[a[:,2]==1,4:], axis=1))
# np.amax(np.amax(a[a[:,2]==1,4:], axis=1))
#   """
#   numberOfWorkers = 4

#   with multiprocessing.Pool(processes=numberOfWorkers) as pool:
#     matches = pool.map(functools.partial(runWorker, numberOfWorkers), list(range(numberOfWorkers)))

#   matches = np.vstack(matches)
#   matches = matches[np.lexsort(matches.T),:]
#   print(len(matches))
#   np.savez_compressed("matches.npyz", matches)



numberOfPatterns = 4
numberOfHalfDays = 14
numberOfPrices = 700



def runWorker(numberOfWorkers, seed, prevPattern, pattern, prices, workerId):
  numberOfSamples = 1000000
  minimumNumberOfMatches = 32 // numberOfWorkers + 1
  if prevPattern is None: prevPattern = 9999
  if pattern is None: pattern = 9999

  cmd = ["./turnips", "sample", "aggregate", str(numberOfSamples), str(minimumNumberOfMatches),
      str(seed + workerId), str(prevPattern), str(pattern)] + [str(x) for x in prices]
  process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
  stdout = process.stdout.decode()

  results = np.reshape(np.fromstring(stdout, sep=" ", dtype=np.int),
      [-1, 1 + 2 * numberOfPatterns + (numberOfHalfDays + 1) * numberOfPrices])
  return results



def computeQuantile(cdf, q):
  tol = 1e-14
  q = min(max(q, tol), 1-tol)
  Idx = np.nonzero(cdf >= q)[0]
  if len(Idx) == 0: print(cdf, q, cdf[-1]-1, q-1)
  return Idx[0]



class DataExtractor(object):
  def __init__(self, data):
    self.data = data
    self.i = 0

  def extract(self, n=None):
    result = self.data[self.i:self.i+n]
    self.i += n
    return result



def wrapText(text, width):
  lines = []
  curLine = ""
  words = text.split()
  appendCurLine = (lambda: lines.append(curLine + ((width - len(curLine)) * " ")))

  for word in words:
    if len("{} {}".format(curLine, word)) > width:
      appendCurLine()
      curLine = ""

    if curLine != "": curLine += " "
    curLine += word

  if curLine != "": appendCurLine()
  return lines



def printAdvice(text, width=40, name="Tom Nook", imageName="tomNook.dat", imageWidth=16):
  text = wrapText(text, width)
  height = max(len(text), 3)
  if height > len(text): text.extend((height - len(text)) * [width * " "])

  bubbleLeftBorder = "/  \n|  \n\\  \n{} \\_".format((height - 3) * " | \n")
  bubbleRightBorder = "   \\\n   |\n  /\n{}_/".format((height - 3) * "  |\n")
  bubbleTopBorder = " ( {} ){}".format(name, (width - len(name) + 1) * "_")
  bubbleBottomBorder = width * "_"

  bubbleLeftBorder = bubbleLeftBorder.splitlines()
  bubbleRightBorder = bubbleRightBorder.splitlines()

  bubble = [bubbleTopBorder]
  bubble.extend(["{}{}{}".format(x, y, z)
      for x, y, z in zip(bubbleLeftBorder[:-1], text, bubbleRightBorder[:-1])])
  bubble.append("{}{}{}".format(bubbleLeftBorder[-1], bubbleBottomBorder, bubbleRightBorder[-1]))

  with open(imageName, "rb") as f: image = f.read().splitlines()
  if len(bubble) > len(image): image = ((len(bubble) - len(image)) * [imageWidth * b" "]) + image
  elif len(bubble) < len(image): bubble = ((len(image) - len(bubble)) * [""]) + bubble

  for imageLine, bubbleLine in zip(image, bubble):
    sys.stdout.buffer.write(imageLine)

    if bubble != "":
      print(" ", end="")
      print(bubbleLine, end="")

    print("")



def formatPercentage(x, article=None):
  x = int(min(max(round(100 * x), 0), 100))
  prefix = ""

  if article == "indefinite":
    prefix = ("an " if (x in [8, 11, 18]) or (80 <= x < 90) else "a ")

  return "{}{:}%".format(prefix, x)



def main():
  numberOfWorkers = 4
  seed = 10 * random.randint(0, 1000000-1)
  pattern = None

  #prevPattern = 2
  #prices = [92, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  #prices = [92, 92, 82, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  #prices = [92, 92, 82, 79, 76, 118, 0, 0, 0, 0, 0, 0, 0, 0]
  #prices = [92, 92, 82, 79, 76, 118, 148, 271, 0, 0, 0, 0, 0, 0]

  prevPattern = 1
  #prices = [92, 92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  #prices = [92, 92, 45, 41, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  #prices = [92, 92, 45, 41, 37, 33, 0, 0, 0, 0, 0, 0, 0, 0]
  prices = [92, 92, 45, 41, 37, 33, 30, 105, 0, 0, 0, 0, 0, 0]

  prices = np.array(prices)
  print("Generating samples... ", end="")

  with multiprocessing.Pool(processes=numberOfWorkers) as pool:
    results = pool.map(functools.partial(runWorker, numberOfWorkers, seed, prevPattern,
        pattern, prices), list(range(numberOfWorkers)))

  results = np.vstack(results).astype(np.float)
  data = DataExtractor(np.sum(results, axis=0))

  numberOfMatches = int(data.extract(1)[0])
  print("done, got {} matches.".format(numberOfMatches))
  print("")

  probPrevPattern = data.extract(4) / numberOfMatches
  probPattern = data.extract(4) / numberOfMatches
  probPrices = (np.reshape(data.extract(numberOfHalfDays * numberOfPrices),
      (numberOfHalfDays, -1)).T / numberOfMatches)
  probMaxPrice = data.extract(numberOfPrices) / numberOfMatches

  t = np.arange(probPrices.shape[1])
  allPrices = np.arange(probPrices.shape[0])
  meanPrices = np.sum(probPrices * allPrices[:,None], axis=0)
  meanMaxPrice = np.sum(probMaxPrice * allPrices)
  cdfPrices = np.cumsum(probPrices, axis=0)
  cdfMaxPrice = np.cumsum(probMaxPrice, axis=0)
  now = np.nonzero(prices != 0)[0][-1]

  qPrices = np.linspace(0.5, 1, 256)
  quantilePrices = np.array([[[computeQuantile(cdfPrices[:,i], (1-q if j == 0 else q))
      for j in range(2)] for i in range(len(t))] for q in qPrices])
  qMaxPrice = np.linspace(0.5, 1, 6)
  quantileMaxPrice = np.array([[computeQuantile(cdfMaxPrice, (1-q if j == 0 else q))
      for j in range(2)] for q in qMaxPrice])

  patternNames = ["Random", "Large Spike", "Decreasing", "Small Spike"]
  formatString = "{:<8} | " + "  ".join(len(patternNames) * ["{:>11}"])
  formatProb = (lambda x: ["{:.1f}%".format(100 * y) for y in x])
  print(formatString.format("Patterns", *patternNames))
  print((8 * "-") + "-+-" + (((11 + 2) * len(patternNames) - 2) * "-"))
  print(formatString.format("Previous", *formatProb(probPrevPattern)))
  print(formatString.format("Current", *formatProb(probPattern)))
  print("")

  probLowerPrice = cdfMaxPrice[prices[now]]

  if probLowerPrice < 0.01:
    advice = "You should wait to sell, hm? Prices will be higher later in the week, yes, yes."
  elif probLowerPrice > 0.99:
    advice = "The prices will only go down from now on, yes, yes. You should sell now, hm?"
  else:
    advice = ("Well, it's hard to say right now, hm? There's {} chance of prices going up "
        "and {} chance of prices going down, yes?").format(
          formatPercentage(1 - probLowerPrice, article="indefinite"),
          formatPercentage(probLowerPrice, article="indefinite"))

  printAdvice(advice)
  print("")

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

  arialRoundedMtRegular = mpl.font_manager.FontProperties(fname="arialRoundedMtRegular.ttf")
  arialRoundedMtBold = mpl.font_manager.FontProperties(fname="arialRoundedMtBold.ttf")

  tt = np.linspace(t[0], t[-1], 1001)
  meanPriceInterp = scipy.interpolate.interp1d(t, meanPrices, kind="linear")(tt)

  xl = [t[0] - 0.5, t[-1] + 0.5]
  yl = [0, 1.05 * np.max(quantilePrices[np.argmin(np.abs(qPrices - 0.95)),:,1])]

  lineColor = "#1f77b4"
  backgroundColor = "#111111"
  foregroundColor = "#eeeeee"
  mixColors = (lambda color1, color2, t: (1 - t) * np.array(mpl.colors.to_rgba(color1)) +
      t * np.array(mpl.colors.to_rgba(color2)))
  lineColor = mixColors(lineColor, "white", 0.5)
  getFillColor = (lambda q: mixColors(lineColor, backgroundColor, 0.3 + 0.65 * (q/ 0.5 - 1)))

  for i in range(qPrices.size - 1, -1, -1):
    quantilePriceInterps = np.array([scipy.interpolate.interp1d(
        t, quantilePrices[i,:,j], kind="linear")(tt) for j in range(2)])

    if i == 0:
      ax3.plot(t, quantilePrices[0,:,0], ".-", color=lineColor)
    else:
      fillColor = getFillColor(qPrices[i])
      ax3.fill(np.concatenate((tt, tt[::-1])), np.concatenate((quantilePriceInterps[0,:],
          quantilePriceInterps[1,::-1])), facecolor=fillColor)

  for i in range(qMaxPrice.size - 1, -1, -1):
    if i == 0:
      ax1.plot([0, 1], 2 * [quantileMaxPrice[0,0]], "-", color=lineColor)
    else:
      fillColor = getFillColor(qMaxPrice[i])
      ax1.fill([0, 1, 1, 0], quantileMaxPrice[i,[0, 0, 1, 1]], facecolor=fillColor)

  ax3.plot(tt, meanPriceInterp, "--", color=lineColor, zorder=1000)
  ax3.plot(t, meanPrices, ".", color=lineColor, zorder=1000)
  ax1.plot([0, 1], 2 * [meanMaxPrice], "--", color=lineColor, zorder=1000)

  for price in [prices[now], *quantileMaxPrice[-1,:]]:
    ax3.plot(xl, 2 * [price], ":", color="#888888")
    ax1.plot([0, 1], 2 * [price], ":", color="#888888")

  ax3.plot(2 * [now], yl, ":", color="#ff6666")

  ax1.set_xlim([0, 1])
  ax2.set_xlim([0, 1])

  ax3.set_xlim(xl)
  ax3.set_ylim(yl)
  ax3.set_title("Turnip Price Forecast (Based on {} Samples)".format(numberOfMatches),
      fontproperties=arialRoundedMtBold, fontsize=14, color=foregroundColor)
  ax3.set_xticks(t)
  ax3.set_xticklabels(["Sun", "", "Mon", "", "Tue", "", "Wed", "", "Thu", "", "Fri", "", "Sat", ""],
      fontproperties=arialRoundedMtRegular, fontsize=12)
  step = 25 * ((yl[1] / 5) // 25)
  yt = np.arange(0, yl[1], step, dtype=np.int)
  ax3.set_yticks(yt)
  ax3.set_yticklabels([])
  ax3.grid(True, color="#666666")

  fig.patch.set_facecolor(backgroundColor)

  for ax in [ax1, ax2, ax3]:
    ax.set_facecolor(backgroundColor)
    for spine in ax.spines.values(): spine.set_color(foregroundColor)
    ax.xaxis.label.set_color(foregroundColor)
    ax.yaxis.label.set_color(foregroundColor)
    ax.tick_params(axis="both", colors=foregroundColor)
    ax.tick_params(axis="y", pad=12)

  ax2.set_facecolor("#ff000000")
  ax2.axes.get_xaxis().set_visible(False)
  ax2.axes.get_yaxis().set_visible(False)
  for spine in ax2.spines.values(): spine.set_visible(False)

  ax1Position = [0.05, 0.1, 0.04, 0.8]
  ax2Position = [ax1Position[0] + ax1Position[2], ax1Position[1], 0.07, ax1Position[3]]
  ax3Position = [ax2Position[0] + ax2Position[2], ax1Position[1], 0.79, ax1Position[3]]
  ax1.set_position(ax1Position)
  ax2.set_position(ax2Position)
  ax3.set_position(ax3Position)

  ax1.set_xticks([])
  ax1.yaxis.tick_right()

  for y in yt:
    ax2.text(0.5, y, str(y), ha="center", va="center", color=foregroundColor,
        fontproperties=arialRoundedMtRegular, fontsize=12)

  trafo = (lambda xy: ax3.transData.inverted().transform(ax3.transAxes.transform(xy)))
  ax3.text(*trafo((0.02, 0.96)), "Bells", rotation=90, ha="left", va="top",
      color=foregroundColor, fontproperties=arialRoundedMtBold, fontsize=12)
  ax3.text(*trafo((0.97, 0.015)), "Time", ha="right", va="bottom",
      color=foregroundColor, fontproperties=arialRoundedMtBold, fontsize=12)
  plt.savefig("/tmp/mafs.png", facecolor=backgroundColor)

  try:
    plt.show()
  except KeyboardInterrupt:
    return



if __name__ == "__main__":
  main()
