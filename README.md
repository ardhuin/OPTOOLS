

# OPTOOLS

processing and simulation tools for optical imagery: looking at ocean waves & currents 
This "OPTOOLS" folder contains miscellaneous simulation / analysis tools of image sequences. 
It was initially set up to investigate anomalies in phase speeds estimated from Sentinel-2 B02-B04 pairs, 
as described in Ardhuin et al. (JGR 2021). 

Some of the matlab code was later modified to contribute to the design of STREAM-O (proposal to ESA EE11). 

F. Ardhuin, LOPS, March 3, 2022


## Contents and quick run: MATLAB

MATLAB folder contains some of the original codes, in particular you can 

I) test the principle of 3-image method using 1D data and Monte-Carlo simulations to check for averaging effect and random errors
opposite_test

II) Simulate some satellite images and analyze the phase speeds

1. define wave spectrum (here using buoy data) and wind and current ... 
[Efth,freq,dir2]=define_spectrum;

U10=6;
d2r=pi/180;
Udir=40.*d2r; % direction to, trig. convention 
Ux=0;
Uy=0;

2. simulate images , for example 

phitrig =[  148.1901  148.8061  149.4561  149.4561];
offspec=[8.9740    9.0674    9.1693    9.1693];
theta=[6.2804    6.2413    6.2114    6.2114];
imgtimes=[0 0.5 1 1];
[imgs,  nx, ny, x, y, dx, dy  ] =   S2_simu(Efth,freq,dir2,U10,Udir,Ux,Uy,imgtimes,offspec,phitrig,theta,10,801 ,1000,0.  ,0.,0.15,1);


3.a 
S2_3ana : this is the 3-image method of Ardhuin et al. (2021) : this is super-slow.

3.b
S2_analysis  : standard 2-image phase difference method

III) Read a piece of Sentinel 2 imagery and do the same kind of analysis

1. load image (here this is just a small piece to make the tar not too big)
load S2img   

2. Run analysis 

2.a 
S2_3ana : this is the 3-image method of Ardhuin et al. (2021) : this is super-slow.


## Contents and quick run: Python 
for running on datarmor you may need to : 
git clone https://gitlab.ifremer.fr/fa1e926/optools OPTOOLS
cd OPTOOLS/PYTHON
module load conda/latest
pip install rasterio
python 

Would be nice to add a few notebooks that do the same things as the matlab script described above .. 


## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.ifremer.fr/fa1e926/optools.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.ifremer.fr/fa1e926/optools/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***



## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.


