# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'w_record_viewer.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_window_record_viewer(object):
    def setupUi(self, window_record_viewer):
        window_record_viewer.setObjectName("window_record_viewer")
        window_record_viewer.resize(930, 542)
        self.centralwidget = QtWidgets.QWidget(window_record_viewer)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(643, 214, 250, 251))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.ButtonExportToExcel = QtWidgets.QPushButton(self.tab_2)
        self.ButtonExportToExcel.setGeometry(QtCore.QRect(660, 550, 75, 23))
        self.ButtonExportToExcel.setObjectName("ButtonExportToExcel")
        self.e_11 = QtWidgets.QRadioButton(self.tab_2)
        self.e_11.setGeometry(QtCore.QRect(53, 71, 16, 17))
        self.e_11.setText("")
        self.e_11.setAutoExclusive(False)
        self.e_11.setObjectName("e_11")
        self.e_14 = QtWidgets.QRadioButton(self.tab_2)
        self.e_14.setGeometry(QtCore.QRect(83, 21, 16, 17))
        self.e_14.setText("")
        self.e_14.setChecked(True)
        self.e_14.setAutoExclusive(False)
        self.e_14.setObjectName("e_14")
        self.e_1 = QtWidgets.QRadioButton(self.tab_2)
        self.e_1.setGeometry(QtCore.QRect(43, 121, 16, 17))
        self.e_1.setText("")
        self.e_1.setAutoExclusive(False)
        self.e_1.setObjectName("e_1")
        self.e_12 = QtWidgets.QRadioButton(self.tab_2)
        self.e_12.setGeometry(QtCore.QRect(73, 51, 16, 17))
        self.e_12.setText("")
        self.e_12.setAutoExclusive(False)
        self.e_12.setObjectName("e_12")
        self.e_9 = QtWidgets.QRadioButton(self.tab_2)
        self.e_9.setGeometry(QtCore.QRect(73, 91, 16, 17))
        self.e_9.setText("")
        self.e_9.setAutoExclusive(False)
        self.e_9.setObjectName("e_9")
        self.e_4 = QtWidgets.QRadioButton(self.tab_2)
        self.e_4.setGeometry(QtCore.QRect(43, 151, 16, 17))
        self.e_4.setText("")
        self.e_4.setAutoExclusive(False)
        self.e_4.setObjectName("e_4")
        self.e_7 = QtWidgets.QRadioButton(self.tab_2)
        self.e_7.setGeometry(QtCore.QRect(43, 181, 16, 17))
        self.e_7.setText("")
        self.e_7.setAutoExclusive(False)
        self.e_7.setObjectName("e_7")
        self.e_0 = QtWidgets.QRadioButton(self.tab_2)
        self.e_0.setGeometry(QtCore.QRect(73, 121, 16, 17))
        self.e_0.setText("")
        self.e_0.setAutoExclusive(False)
        self.e_0.setObjectName("e_0")
        self.e_3 = QtWidgets.QRadioButton(self.tab_2)
        self.e_3.setGeometry(QtCore.QRect(73, 151, 16, 17))
        self.e_3.setText("")
        self.e_3.setAutoExclusive(False)
        self.e_3.setObjectName("e_3")
        self.e_15 = QtWidgets.QRadioButton(self.tab_2)
        self.e_15.setGeometry(QtCore.QRect(103, 21, 16, 17))
        self.e_15.setText("")
        self.e_15.setAutoExclusive(False)
        self.e_15.setObjectName("e_15")
        self.e_13 = QtWidgets.QRadioButton(self.tab_2)
        self.e_13.setGeometry(QtCore.QRect(103, 51, 16, 17))
        self.e_13.setText("")
        self.e_13.setAutoExclusive(False)
        self.e_13.setObjectName("e_13")
        self.e_8 = QtWidgets.QRadioButton(self.tab_2)
        self.e_8.setGeometry(QtCore.QRect(103, 121, 16, 17))
        self.e_8.setText("")
        self.e_8.setAutoExclusive(False)
        self.e_8.setObjectName("e_8")
        self.e_10 = QtWidgets.QRadioButton(self.tab_2)
        self.e_10.setGeometry(QtCore.QRect(103, 91, 16, 17))
        self.e_10.setText("")
        self.e_10.setAutoExclusive(False)
        self.e_10.setObjectName("e_10")
        self.e_2 = QtWidgets.QRadioButton(self.tab_2)
        self.e_2.setGeometry(QtCore.QRect(103, 151, 16, 17))
        self.e_2.setText("")
        self.e_2.setAutoExclusive(False)
        self.e_2.setObjectName("e_2")
        self.e_6 = QtWidgets.QRadioButton(self.tab_2)
        self.e_6.setGeometry(QtCore.QRect(73, 181, 16, 17))
        self.e_6.setText("")
        self.e_6.setAutoExclusive(False)
        self.e_6.setObjectName("e_6")
        self.e_5 = QtWidgets.QRadioButton(self.tab_2)
        self.e_5.setGeometry(QtCore.QRect(103, 181, 16, 17))
        self.e_5.setText("")
        self.e_5.setAutoExclusive(False)
        self.e_5.setObjectName("e_5")
        self.e_29 = QtWidgets.QRadioButton(self.tab_2)
        self.e_29.setGeometry(QtCore.QRect(133, 151, 16, 17))
        self.e_29.setText("")
        self.e_29.setAutoExclusive(False)
        self.e_29.setObjectName("e_29")
        self.e_18 = QtWidgets.QRadioButton(self.tab_2)
        self.e_18.setGeometry(QtCore.QRect(133, 51, 16, 17))
        self.e_18.setText("")
        self.e_18.setAutoExclusive(False)
        self.e_18.setObjectName("e_18")
        self.e_21 = QtWidgets.QRadioButton(self.tab_2)
        self.e_21.setGeometry(QtCore.QRect(133, 91, 16, 17))
        self.e_21.setText("")
        self.e_21.setAutoExclusive(False)
        self.e_21.setObjectName("e_21")
        self.e_23 = QtWidgets.QRadioButton(self.tab_2)
        self.e_23.setGeometry(QtCore.QRect(133, 121, 16, 17))
        self.e_23.setText("")
        self.e_23.setAutoExclusive(False)
        self.e_23.setObjectName("e_23")
        self.e_16 = QtWidgets.QRadioButton(self.tab_2)
        self.e_16.setGeometry(QtCore.QRect(133, 21, 16, 17))
        self.e_16.setText("")
        self.e_16.setAutoExclusive(False)
        self.e_16.setObjectName("e_16")
        self.e_26 = QtWidgets.QRadioButton(self.tab_2)
        self.e_26.setGeometry(QtCore.QRect(133, 181, 16, 17))
        self.e_26.setText("")
        self.e_26.setAutoExclusive(False)
        self.e_26.setObjectName("e_26")
        self.e_28 = QtWidgets.QRadioButton(self.tab_2)
        self.e_28.setGeometry(QtCore.QRect(163, 151, 16, 17))
        self.e_28.setText("")
        self.e_28.setAutoExclusive(False)
        self.e_28.setObjectName("e_28")
        self.e_31 = QtWidgets.QRadioButton(self.tab_2)
        self.e_31.setGeometry(QtCore.QRect(163, 121, 16, 17))
        self.e_31.setText("")
        self.e_31.setObjectName("e_31")
        self.e_17 = QtWidgets.QRadioButton(self.tab_2)
        self.e_17.setGeometry(QtCore.QRect(153, 21, 16, 17))
        self.e_17.setToolTipDuration(10)
        self.e_17.setWhatsThis("")
        self.e_17.setAccessibleName("")
        self.e_17.setAccessibleDescription("")
        self.e_17.setText("")
        self.e_17.setAutoExclusive(False)
        self.e_17.setObjectName("e_17")
        self.e_19 = QtWidgets.QRadioButton(self.tab_2)
        self.e_19.setGeometry(QtCore.QRect(163, 51, 16, 17))
        self.e_19.setText("")
        self.e_19.setAutoExclusive(False)
        self.e_19.setObjectName("e_19")
        self.e_25 = QtWidgets.QRadioButton(self.tab_2)
        self.e_25.setGeometry(QtCore.QRect(163, 181, 16, 17))
        self.e_25.setText("")
        self.e_25.setAutoExclusive(False)
        self.e_25.setObjectName("e_25")
        self.e_22 = QtWidgets.QRadioButton(self.tab_2)
        self.e_22.setGeometry(QtCore.QRect(163, 91, 16, 17))
        self.e_22.setText("")
        self.e_22.setAutoExclusive(False)
        self.e_22.setObjectName("e_22")
        self.e_27 = QtWidgets.QRadioButton(self.tab_2)
        self.e_27.setGeometry(QtCore.QRect(193, 151, 16, 17))
        self.e_27.setText("")
        self.e_27.setAutoExclusive(False)
        self.e_27.setObjectName("e_27")
        self.e_30 = QtWidgets.QRadioButton(self.tab_2)
        self.e_30.setGeometry(QtCore.QRect(193, 121, 16, 17))
        self.e_30.setText("")
        self.e_30.setAutoExclusive(False)
        self.e_30.setObjectName("e_30")
        self.e_20 = QtWidgets.QRadioButton(self.tab_2)
        self.e_20.setGeometry(QtCore.QRect(183, 71, 16, 17))
        self.e_20.setText("")
        self.e_20.setAutoExclusive(False)
        self.e_20.setObjectName("e_20")
        self.e_24 = QtWidgets.QRadioButton(self.tab_2)
        self.e_24.setGeometry(QtCore.QRect(193, 181, 16, 17))
        self.e_24.setText("")
        self.e_24.setAutoExclusive(False)
        self.e_24.setObjectName("e_24")
        self.frame_3 = QtWidgets.QFrame(self.tab_2)
        self.frame_3.setGeometry(QtCore.QRect(33, 11, 181, 191))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.frame_3.raise_()
        self.ButtonExportToExcel.raise_()
        self.e_11.raise_()
        self.e_14.raise_()
        self.e_1.raise_()
        self.e_12.raise_()
        self.e_9.raise_()
        self.e_4.raise_()
        self.e_7.raise_()
        self.e_0.raise_()
        self.e_3.raise_()
        self.e_15.raise_()
        self.e_13.raise_()
        self.e_8.raise_()
        self.e_10.raise_()
        self.e_2.raise_()
        self.e_6.raise_()
        self.e_5.raise_()
        self.e_29.raise_()
        self.e_18.raise_()
        self.e_21.raise_()
        self.e_23.raise_()
        self.e_16.raise_()
        self.e_26.raise_()
        self.e_28.raise_()
        self.e_31.raise_()
        self.e_17.raise_()
        self.e_19.raise_()
        self.e_25.raise_()
        self.e_22.raise_()
        self.e_27.raise_()
        self.e_30.raise_()
        self.e_20.raise_()
        self.e_24.raise_()
        self.tabWidget.addTab(self.tab_2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.frame_5 = QtWidgets.QFrame(self.tab)
        self.frame_5.setGeometry(QtCore.QRect(30, 12, 181, 191))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.et_2 = QtWidgets.QRadioButton(self.frame_5)
        self.et_2.setGeometry(QtCore.QRect(100, 40, 16, 17))
        self.et_2.setText("")
        self.et_2.setAutoExclusive(False)
        self.et_2.setObjectName("et_2")
        self.et_5 = QtWidgets.QRadioButton(self.frame_5)
        self.et_5.setGeometry(QtCore.QRect(130, 170, 16, 17))
        self.et_5.setText("")
        self.et_5.setAutoExclusive(False)
        self.et_5.setObjectName("et_5")
        self.et_11 = QtWidgets.QRadioButton(self.frame_5)
        self.et_11.setGeometry(QtCore.QRect(40, 140, 16, 17))
        self.et_11.setText("")
        self.et_11.setAutoExclusive(False)
        self.et_11.setObjectName("et_11")
        self.et_6 = QtWidgets.QRadioButton(self.tab)
        self.et_6.setGeometry(QtCore.QRect(160, 92, 16, 17))
        self.et_6.setText("")
        self.et_6.setAutoExclusive(False)
        self.et_6.setObjectName("et_6")
        self.et_12 = QtWidgets.QRadioButton(self.tab)
        self.et_12.setGeometry(QtCore.QRect(70, 182, 16, 17))
        self.et_12.setText("")
        self.et_12.setAutoExclusive(False)
        self.et_12.setObjectName("et_12")
        self.et_13 = QtWidgets.QRadioButton(self.tab)
        self.et_13.setGeometry(QtCore.QRect(70, 92, 16, 17))
        self.et_13.setText("")
        self.et_13.setAutoExclusive(False)
        self.et_13.setObjectName("et_13")
        self.et_0 = QtWidgets.QRadioButton(self.tab)
        self.et_0.setGeometry(QtCore.QRect(160, 122, 16, 17))
        self.et_0.setText("")
        self.et_0.setObjectName("et_0")
        self.et_3 = QtWidgets.QRadioButton(self.tab)
        self.et_3.setGeometry(QtCore.QRect(160, 52, 16, 17))
        self.et_3.setText("")
        self.et_3.setAutoExclusive(False)
        self.et_3.setObjectName("et_3")
        self.et_8 = QtWidgets.QRadioButton(self.tab)
        self.et_8.setGeometry(QtCore.QRect(100, 22, 16, 17))
        self.et_8.setText("")
        self.et_8.setAutoExclusive(False)
        self.et_8.setObjectName("et_8")
        self.et_4 = QtWidgets.QRadioButton(self.tab)
        self.et_4.setGeometry(QtCore.QRect(160, 152, 16, 17))
        self.et_4.setText("")
        self.et_4.setAutoExclusive(False)
        self.et_4.setObjectName("et_4")
        self.et_10 = QtWidgets.QRadioButton(self.tab)
        self.et_10.setGeometry(QtCore.QRect(70, 52, 16, 17))
        self.et_10.setText("")
        self.et_10.setAutoExclusive(False)
        self.et_10.setObjectName("et_10")
        self.et_7 = QtWidgets.QRadioButton(self.tab)
        self.et_7.setGeometry(QtCore.QRect(70, 122, 16, 17))
        self.et_7.setText("")
        self.et_7.setAutoExclusive(False)
        self.et_7.setObjectName("et_7")
        self.et_1 = QtWidgets.QRadioButton(self.tab)
        self.et_1.setGeometry(QtCore.QRect(130, 22, 16, 17))
        self.et_1.setText("")
        self.et_1.setAutoExclusive(False)
        self.et_1.setObjectName("et_1")
        self.et_9 = QtWidgets.QRadioButton(self.tab)
        self.et_9.setGeometry(QtCore.QRect(100, 52, 16, 17))
        self.et_9.setText("")
        self.et_9.setAutoExclusive(False)
        self.et_9.setObjectName("et_9")
        self.et_14 = QtWidgets.QRadioButton(self.tab)
        self.et_14.setGeometry(QtCore.QRect(215, 182, 16, 17))
        self.et_14.setText("")
        self.et_14.setAutoExclusive(False)
        self.et_14.setObjectName("et_14")
        self.et_15 = QtWidgets.QRadioButton(self.tab)
        self.et_15.setGeometry(QtCore.QRect(13, 183, 16, 17))
        self.et_15.setText("")
        self.et_15.setAutoExclusive(False)
        self.et_15.setObjectName("et_15")
        self.tabWidget.addTab(self.tab, "")
        self.frame_Experimental_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_Experimental_2.setGeometry(QtCore.QRect(0, 0, 701, 81))
        self.frame_Experimental_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_Experimental_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_Experimental_2.setObjectName("frame_Experimental_2")
        self.ButtonAdd_IndRecords = QtWidgets.QPushButton(self.frame_Experimental_2)
        self.ButtonAdd_IndRecords.setGeometry(QtCore.QRect(10, 9, 111, 31))
        self.ButtonAdd_IndRecords.setObjectName("ButtonAdd_IndRecords")
        self.ButtonDelete_IndRecords = QtWidgets.QPushButton(self.frame_Experimental_2)
        self.ButtonDelete_IndRecords.setGeometry(QtCore.QRect(9, 41, 112, 31))
        self.ButtonDelete_IndRecords.setObjectName("ButtonDelete_IndRecords")
        self.listWidget_Indiv_recordings = QtWidgets.QListWidget(self.frame_Experimental_2)
        self.listWidget_Indiv_recordings.setGeometry(QtCore.QRect(120, 10, 571, 61))
        self.listWidget_Indiv_recordings.setObjectName("listWidget_Indiv_recordings")
        self.ButtonLoadRecordings = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonLoadRecordings.setGeometry(QtCore.QRect(710, 40, 151, 41))
        self.ButtonLoadRecordings.setObjectName("ButtonLoadRecordings")
        self.ScrollBarCurrentRecord = QtWidgets.QScrollBar(self.centralwidget)
        self.ScrollBarCurrentRecord.setGeometry(QtCore.QRect(20, 84, 81, 31))
        self.ScrollBarCurrentRecord.setOrientation(QtCore.Qt.Horizontal)
        self.ScrollBarCurrentRecord.setObjectName("ScrollBarCurrentRecord")
        self.labelCurrentRecording = QtWidgets.QLabel(self.centralwidget)
        self.labelCurrentRecording.setGeometry(QtCore.QRect(110, 90, 751, 31))
        self.labelCurrentRecording.setText("")
        self.labelCurrentRecording.setObjectName("labelCurrentRecording")
        self.BoxBinSize = QtWidgets.QSpinBox(self.centralwidget)
        self.BoxBinSize.setGeometry(QtCore.QRect(10, 150, 71, 22))
        self.BoxBinSize.setMaximum(600000)
        self.BoxBinSize.setSingleStep(30)
        self.BoxBinSize.setProperty("value", 30)
        self.BoxBinSize.setObjectName("BoxBinSize")
        self.ButtonCoherogram = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonCoherogram.setEnabled(False)
        self.ButtonCoherogram.setGeometry(QtCore.QRect(233, 240, 91, 41))
        self.ButtonCoherogram.setObjectName("ButtonCoherogram")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(87, 150, 151, 16))
        self.label_5.setObjectName("label_5")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(98, 184, 47, 16))
        self.label_8.setObjectName("label_8")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(98, 216, 47, 13))
        self.label_22.setObjectName("label_22")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setEnabled(False)
        self.label_25.setGeometry(QtCore.QRect(10, 250, 211, 16))
        self.label_25.setObjectName("label_25")
        self.ButtonPlotRawData = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonPlotRawData.setGeometry(QtCore.QRect(223, 140, 101, 41))
        self.ButtonPlotRawData.setObjectName("ButtonPlotRawData")
        self.ButtonPlotPS = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonPlotPS.setGeometry(QtCore.QRect(223, 190, 101, 41))
        self.ButtonPlotPS.setObjectName("ButtonPlotPS")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(330, 140, 141, 20))
        self.label_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_4.setObjectName("label_4")
        self.ButtonExportIndPNG = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonExportIndPNG.setGeometry(QtCore.QRect(415, 170, 40, 31))
        self.ButtonExportIndPNG.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.ButtonExportIndPNG.setObjectName("ButtonExportIndPNG")
        self.ButtonExportIndPDF = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonExportIndPDF.setGeometry(QtCore.QRect(355, 170, 40, 31))
        self.ButtonExportIndPDF.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.ButtonExportIndPDF.setObjectName("ButtonExportIndPDF")
        self.ButtonCloseFigures = QtWidgets.QPushButton(self.centralwidget)
        self.ButtonCloseFigures.setGeometry(QtCore.QRect(345, 210, 121, 30))
        self.ButtonCloseFigures.setStyleSheet("background-color: rgb(255, 0, 0);")
        self.ButtonCloseFigures.setObjectName("ButtonCloseFigures")
        self.label_Tmax = QtWidgets.QLabel(self.centralwidget)
        self.label_Tmax.setGeometry(QtCore.QRect(20, 119, 71, 16))
        self.label_Tmax.setText("")
        self.label_Tmax.setObjectName("label_Tmax")
        self.radioSelectLeftHem = QtWidgets.QRadioButton(self.centralwidget)
        self.radioSelectLeftHem.setGeometry(QtCore.QRect(770, 161, 111, 17))
        self.radioSelectLeftHem.setObjectName("radioSelectLeftHem")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(678, 130, 171, 20))
        self.label_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_7.setObjectName("label_7")
        self.radioSelectNone = QtWidgets.QRadioButton(self.centralwidget)
        self.radioSelectNone.setGeometry(QtCore.QRect(639, 181, 131, 17))
        self.radioSelectNone.setObjectName("radioSelectNone")
        self.radioSelectAll = QtWidgets.QRadioButton(self.centralwidget)
        self.radioSelectAll.setGeometry(QtCore.QRect(639, 161, 121, 17))
        self.radioSelectAll.setObjectName("radioSelectAll")
        self.radioSelectRightHem = QtWidgets.QRadioButton(self.centralwidget)
        self.radioSelectRightHem.setGeometry(QtCore.QRect(770, 181, 121, 17))
        self.radioSelectRightHem.setObjectName("radioSelectRightHem")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(627, 153, 281, 51))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 312, 111, 20))
        self.label_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_6.setObjectName("label_6")
        self.BoxHighFreq = QtWidgets.QSpinBox(self.centralwidget)
        self.BoxHighFreq.setGeometry(QtCore.QRect(20, 380, 71, 22))
        self.BoxHighFreq.setMaximum(1024)
        self.BoxHighFreq.setSingleStep(30)
        self.BoxHighFreq.setProperty("value", 1024)
        self.BoxHighFreq.setObjectName("BoxHighFreq")
        self.BoxLowFreq = QtWidgets.QSpinBox(self.centralwidget)
        self.BoxLowFreq.setGeometry(QtCore.QRect(20, 350, 71, 22))
        self.BoxLowFreq.setMaximum(1024)
        self.BoxLowFreq.setSingleStep(60)
        self.BoxLowFreq.setObjectName("BoxLowFreq")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(100, 350, 61, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(100, 380, 151, 21))
        self.label_10.setObjectName("label_10")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 340, 151, 110))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.ButtonApplyBandFilter = QtWidgets.QPushButton(self.frame_2)
        self.ButtonApplyBandFilter.setGeometry(QtCore.QRect(10, 70, 101, 31))
        self.ButtonApplyBandFilter.setObjectName("ButtonApplyBandFilter")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(807, 13, 111, 21))
        self.label_12.setObjectName("label_12")
        self.SpinBoxSamplingRate = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.SpinBoxSamplingRate.setGeometry(QtCore.QRect(710, 10, 91, 24))
        self.SpinBoxSamplingRate.setDecimals(3)
        self.SpinBoxSamplingRate.setMaximum(2000.0)
        self.SpinBoxSamplingRate.setProperty("value", 125.205)
        self.SpinBoxSamplingRate.setObjectName("SpinBoxSamplingRate")
        self.e_32 = QtWidgets.QRadioButton(self.centralwidget)
        self.e_32.setGeometry(QtCore.QRect(419, 434, 16, 17))
        self.e_32.setText("")
        self.e_32.setAutoExclusive(False)
        self.e_32.setObjectName("e_32")
        self.e_export_amplitude = QtWidgets.QRadioButton(self.centralwidget)
        self.e_export_amplitude.setGeometry(QtCore.QRect(419, 383, 16, 17))
        self.e_export_amplitude.setText("")
        self.e_export_amplitude.setAutoExclusive(False)
        self.e_export_amplitude.setObjectName("e_export_amplitude")
        self.e_export_bandpass = QtWidgets.QRadioButton(self.centralwidget)
        self.e_export_bandpass.setGeometry(QtCore.QRect(419, 408, 16, 17))
        self.e_export_bandpass.setText("")
        self.e_export_bandpass.setAutoExclusive(False)
        self.e_export_bandpass.setObjectName("e_export_bandpass")
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(401, 337, 201, 171))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.label_20 = QtWidgets.QLabel(self.frame_4)
        self.label_20.setEnabled(True)
        self.label_20.setGeometry(QtCore.QRect(61, 12, 141, 20))
        self.label_20.setObjectName("label_20")
        self.spinBoxDownsampl = QtWidgets.QSpinBox(self.frame_4)
        self.spinBoxDownsampl.setEnabled(True)
        self.spinBoxDownsampl.setGeometry(QtCore.QRect(11, 12, 41, 20))
        self.spinBoxDownsampl.setMinimum(1)
        self.spinBoxDownsampl.setMaximum(128)
        self.spinBoxDownsampl.setProperty("value", 1)
        self.spinBoxDownsampl.setObjectName("spinBoxDownsampl")
        self.label_21 = QtWidgets.QLabel(self.frame_4)
        self.label_21.setEnabled(True)
        self.label_21.setGeometry(QtCore.QRect(43, 43, 151, 20))
        self.label_21.setObjectName("label_21")
        self.label_23 = QtWidgets.QLabel(self.frame_4)
        self.label_23.setEnabled(True)
        self.label_23.setGeometry(QtCore.QRect(43, 68, 141, 20))
        self.label_23.setObjectName("label_23")
        self.ButtonExportFile = QtWidgets.QPushButton(self.frame_4)
        self.ButtonExportFile.setGeometry(QtCore.QRect(20, 130, 101, 31))
        self.ButtonExportFile.setObjectName("ButtonExportFile")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(167, 312, 171, 20))
        self.label_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_13.setObjectName("label_13")
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(166, 340, 171, 81))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.ButtonApplyBandFilter_2 = QtWidgets.QPushButton(self.frame_6)
        self.ButtonApplyBandFilter_2.setGeometry(QtCore.QRect(10, 39, 101, 31))
        self.ButtonApplyBandFilter_2.setObjectName("ButtonApplyBandFilter_2")
        self.label_15 = QtWidgets.QLabel(self.frame_6)
        self.label_15.setGeometry(QtCore.QRect(89, 10, 61, 16))
        self.label_15.setObjectName("label_15")
        self.BoxLowFreq_2 = QtWidgets.QSpinBox(self.frame_6)
        self.BoxLowFreq_2.setGeometry(QtCore.QRect(9, 10, 71, 22))
        self.BoxLowFreq_2.setMaximum(1024)
        self.BoxLowFreq_2.setSingleStep(60)
        self.BoxLowFreq_2.setObjectName("BoxLowFreq_2")
        self.BoxFilterCores = QtWidgets.QSpinBox(self.centralwidget)
        self.BoxFilterCores.setGeometry(QtCore.QRect(483, 168, 40, 22))
        self.BoxFilterCores.setMaximum(48)
        self.BoxFilterCores.setSingleStep(60)
        self.BoxFilterCores.setProperty("value", 1)
        self.BoxFilterCores.setObjectName("BoxFilterCores")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(533, 169, 50, 16))
        self.label_11.setObjectName("label_11")
        self.radioUseGPUfiltering = QtWidgets.QRadioButton(self.centralwidget)
        self.radioUseGPUfiltering.setGeometry(QtCore.QRect(484, 200, 101, 17))
        self.radioUseGPUfiltering.setObjectName("radioUseGPUfiltering")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(474, 140, 141, 20))
        self.label_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_14.setObjectName("label_14")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(460, 311, 111, 20))
        self.label_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_16.setObjectName("label_16")
        self.BoxTmin = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.BoxTmin.setGeometry(QtCore.QRect(11, 180, 81, 24))
        self.BoxTmin.setDecimals(1)
        self.BoxTmin.setMaximum(1000000.0)
        self.BoxTmin.setObjectName("BoxTmin")
        self.BoxTmax = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.BoxTmax.setGeometry(QtCore.QRect(10, 210, 81, 24))
        self.BoxTmax.setDecimals(1)
        self.BoxTmax.setMaximum(1000000.0)
        self.BoxTmax.setProperty("value", 1.0)
        self.BoxTmax.setObjectName("BoxTmax")
        self.frame_4.raise_()
        self.frame_2.raise_()
        self.frame.raise_()
        self.tabWidget.raise_()
        self.frame_Experimental_2.raise_()
        self.ButtonLoadRecordings.raise_()
        self.ScrollBarCurrentRecord.raise_()
        self.labelCurrentRecording.raise_()
        self.BoxBinSize.raise_()
        self.ButtonCoherogram.raise_()
        self.label_5.raise_()
        self.label_8.raise_()
        self.label_22.raise_()
        self.label_25.raise_()
        self.ButtonPlotRawData.raise_()
        self.ButtonPlotPS.raise_()
        self.label_4.raise_()
        self.ButtonExportIndPNG.raise_()
        self.ButtonExportIndPDF.raise_()
        self.ButtonCloseFigures.raise_()
        self.label_Tmax.raise_()
        self.radioSelectLeftHem.raise_()
        self.label_7.raise_()
        self.radioSelectNone.raise_()
        self.radioSelectAll.raise_()
        self.radioSelectRightHem.raise_()
        self.label_6.raise_()
        self.BoxHighFreq.raise_()
        self.BoxLowFreq.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.label_12.raise_()
        self.SpinBoxSamplingRate.raise_()
        self.e_32.raise_()
        self.e_export_amplitude.raise_()
        self.e_export_bandpass.raise_()
        self.label_13.raise_()
        self.frame_6.raise_()
        self.BoxFilterCores.raise_()
        self.label_11.raise_()
        self.radioUseGPUfiltering.raise_()
        self.label_14.raise_()
        self.label_16.raise_()
        self.BoxTmin.raise_()
        self.BoxTmax.raise_()
        window_record_viewer.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(window_record_viewer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 930, 26))
        self.menubar.setObjectName("menubar")
        window_record_viewer.setMenuBar(self.menubar)

        self.retranslateUi(window_record_viewer)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(window_record_viewer)

    def retranslateUi(self, window_record_viewer):
        _translate = QtCore.QCoreApplication.translate
        window_record_viewer.setWindowTitle(_translate("window_record_viewer", "MainWindow"))
        self.ButtonExportToExcel.setText(_translate("window_record_viewer", "Export"))
        self.e_11.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1FL_S1DZ_LEFT</p></body></html>"))
        self.e_14.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M1_FrA_LEFT</p></body></html>"))
        self.e_1.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1DZ_S1Bf_LEFT</p></body></html>"))
        self.e_12.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M1_ant_LEFT</p></body></html>"))
        self.e_9.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1HL_S1FL_LEFT</p></body></html>"))
        self.e_4.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2L_LEFT</p></body></html>"))
        self.e_7.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2L_V1B_LEFT</p></body></html>"))
        self.e_0.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1Tr_LEFT</p></body></html>"))
        self.e_3.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2ML_LEFT</p></body></html>"))
        self.e_15.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M2_FrA_LEFT</p></body></html>"))
        self.e_13.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M2_ant_LEFT</p></body></html>"))
        self.e_8.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M2_post_LEFT</p></body></html>"))
        self.e_10.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M1_post_LEFT</p></body></html>"))
        self.e_2.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2MM_RSA_LEFT</p></body></html>"))
        self.e_6.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V1M_LEFT</p></body></html>"))
        self.e_5.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2MM_LEFT</p></body></html>"))
        self.e_29.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2MM_RSA_RIGHT</p></body></html>"))
        self.e_18.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M2_ant_RIGHT</p></body></html>"))
        self.e_21.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M1_post_RIGHT</p></body></html>"))
        self.e_23.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M2_post_RIGHT</p></body></html>"))
        self.e_16.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M2_FrA_RIGHT</p></body></html>"))
        self.e_26.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2MM_RIGHT</p></body></html>"))
        self.e_28.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2ML_RIGHT</p></body></html>"))
        self.e_31.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1Tr_RIGHT</p></body></html>"))
        self.e_17.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M1_FrA_RIGHT</p></body></html>"))
        self.e_17.setStatusTip(_translate("window_record_viewer", "M1_FrA_RIGHT"))
        self.e_19.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M1_ant_RIGHT</p></body></html>"))
        self.e_25.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V1M_RIGHT</p></body></html>"))
        self.e_22.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1HL_S1FL_RIGHT</p></body></html>"))
        self.e_27.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2L_RIGHT</p></body></html>"))
        self.e_30.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1DZ_S1Bf_RIGHT</p></body></html>"))
        self.e_20.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1FL_S1DZ_RIGHT</p></body></html>"))
        self.e_24.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2L_V1B_RIGHT</p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("window_record_viewer", "32 OpenEphys"))
        self.et_2.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M2_ant</p></body></html>"))
        self.et_5.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V1M</p></body></html>"))
        self.et_6.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1HL_S1FL</p></body></html>"))
        self.et_0.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>S1Tr</p></body></html>"))
        self.et_3.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M1_ant</p></body></html>"))
        self.et_4.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>V2ML</p></body></html>"))
        self.et_1.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>M2_FrA</p></body></html>"))
        self.et_14.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>EMG_Right</p></body></html>"))
        self.et_15.setToolTip(_translate("window_record_viewer", "<html><head/><body><p>EMG_Left</p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("window_record_viewer", "16 Taini"))
        self.ButtonAdd_IndRecords.setText(_translate("window_record_viewer", "Add folder"))
        self.ButtonDelete_IndRecords.setText(_translate("window_record_viewer", "Delete"))
        self.ButtonLoadRecordings.setText(_translate("window_record_viewer", "Load recordings"))
        self.ButtonCoherogram.setText(_translate("window_record_viewer", "Wavelet"))
        self.label_5.setText(_translate("window_record_viewer", "Bin to plot (s)"))
        self.label_8.setText(_translate("window_record_viewer", "Tmin"))
        self.label_22.setText(_translate("window_record_viewer", "Tmax"))
        self.label_25.setText(_translate("window_record_viewer", "If one channel selected"))
        self.ButtonPlotRawData.setText(_translate("window_record_viewer", "Plot Raw"))
        self.ButtonPlotPS.setText(_translate("window_record_viewer", "Plot PS"))
        self.label_4.setText(_translate("window_record_viewer", "<html><head/><body><p align=\"center\">Export Figures</p></body></html>"))
        self.ButtonExportIndPNG.setText(_translate("window_record_viewer", "PNG"))
        self.ButtonExportIndPDF.setText(_translate("window_record_viewer", "PDF"))
        self.ButtonCloseFigures.setText(_translate("window_record_viewer", "Close Figures"))
        self.radioSelectLeftHem.setText(_translate("window_record_viewer", "Left Hem"))
        self.label_7.setText(_translate("window_record_viewer", "<html><head/><body><p align=\"center\">Channels to show</p></body></html>"))
        self.radioSelectNone.setText(_translate("window_record_viewer", "Select none"))
        self.radioSelectAll.setText(_translate("window_record_viewer", "Select all"))
        self.radioSelectRightHem.setText(_translate("window_record_viewer", "Right Hem"))
        self.label_6.setText(_translate("window_record_viewer", "<html><head/><body><p align=\"center\">Band Filter</p></body></html>"))
        self.label_9.setText(_translate("window_record_viewer", "low f"))
        self.label_10.setText(_translate("window_record_viewer", "high f"))
        self.ButtonApplyBandFilter.setText(_translate("window_record_viewer", "Apply filter"))
        self.label_12.setText(_translate("window_record_viewer", "Sampling R"))
        self.label_20.setText(_translate("window_record_viewer", "Downsampling"))
        self.label_21.setText(_translate("window_record_viewer", "Amplitude filter"))
        self.label_23.setText(_translate("window_record_viewer", "Band Pass filter"))
        self.ButtonExportFile.setText(_translate("window_record_viewer", "Export file"))
        self.label_13.setText(_translate("window_record_viewer", "<html><head/><body><p align=\"center\">Amplitude Filter</p></body></html>"))
        self.ButtonApplyBandFilter_2.setText(_translate("window_record_viewer", "Apply filter"))
        self.label_15.setText(_translate("window_record_viewer", "thresh"))
        self.label_11.setText(_translate("window_record_viewer", "cores"))
        self.radioUseGPUfiltering.setText(_translate("window_record_viewer", "use GPU"))
        self.label_14.setText(_translate("window_record_viewer", "<html><head/><body><p align=\"center\">Parallelization</p></body></html>"))
        self.label_16.setText(_translate("window_record_viewer", "<html><head/><body><p align=\"center\">Export file</p></body></html>"))

