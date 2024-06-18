library(WRS)

df = read.csv("~/Documents/studentsRAPostdocs/nolan/catgen/git/generating_categories_cbb/Experiments/middle-bottom/analysis/yranges.txt")

bottomYs = df$yrange[df$condition == "Bottom"] 
middleYs = df$yrange[df$condition == "Middle"] 

normFailTestBot = shapiro.test(bottomYs)
normFailTestMid = shapiro.test(middleYs)

wilcoxTest =  wilcox.test(bottomYs,middleYs)
yuenTestVal = yuen(bottomYs,middleYs,alpha=.05)
yuenBTTestVal = yuenbt(bottomYs,middleYs,nboot=2000,side=F)
