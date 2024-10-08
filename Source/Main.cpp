/*
	Copyright (c) 2011, T. Kroes <t.kroes@tudelft.nl>
	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
	- Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "Stable.h"

#include "MainWindow.h"

int main1(int ArgumentCount, char* pArgv[])
{
	// Create the application
    QApplication Application(ArgumentCount, pArgv);
	// Adjust style
	Application.setStyle("plastique");
	Application.setOrganizationName("TU Delft");
	Application.setApplicationName("Exposure Render");
	
	// Application settings
	QSettings Settings;
	std::cout << "VTK_VERSION:\t" << VTK_VERSION << std::endl;
	std::cout << "QT_VERSION:\t" << QT_VERSION_MAJOR <<"."<<QT_VERSION_MINOR << "." << QT_VERSION_PATCH << std::endl;
	std::cout << "VS_VERSION:\t" << _MSC_VER << std::endl;
	std::cout << "CUDA_VERSION:\t" << CUDA_VERSION << std::endl;
	Settings.setValue("version", "1.0.0");

	// Main window
	CMainWindow MainWindow;

	// Show the main window
	//gpMainWindow = &MainWindow;

	// Show it
	MainWindow.show();

	MainWindow.setWindowIcon(GetIcon("grid"));

	

	// Load default presets
	gStatus.SetLoadPreset("Default");

//	Log("Device memory: " + QString::number(GetUsedCudaMemory() / MB, 'f', 2) + "/" + QString::number(GetTotalCudaMemory() / MB, 'f', 2) + " MB", "memory");

	// Override the application setting to enforce the display of the startup dialog
	Settings.setValue("startup/dialog/show", QVariant(true));

	// Show startup dialog
	if (Settings.value("startup/dialog/show").toBool() == true)
		MainWindow.ShowStartupDialog();

	// Execute the application
	int Result = Application.exec();

	return Result;
}