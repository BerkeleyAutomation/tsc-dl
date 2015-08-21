function bg_Image(Image_file,Fig_position,reduc_fact,figHandle)
    % legend_Image(Image_file,Fig_position,reduc_fact)
    % Image_file= Image File name (full)
    % Fig_position: position on figure [1 2 3 4]= [U/L U/R D/R D/L]
    % reduc_fact : Ratio Image_size/Figure/size

    %% Figure example
    % hist(rand(1,2000),100);
    Dim1=get(figHandle,'position');
    L =Dim1(1); D=Dim1(2); W=Dim1(3); H=Dim1(4);
    %%
    % Calculate the Image position on figure
    % reduction factor of the size
    im_W=W/reduc_fact;
    im_H=H/reduc_fact;

    switch Fig_position
        case 1 % Position 1 : Upper/Left
    im_L=L;
    im_D=D+H-im_H;
        case 2 % Position 2 : Upper/Right
    im_L=L+W-im_W;
    im_D=D+H-im_H;
        case 3 % Position 3 : Down/Right
    im_L=L+W-im_W;
    im_D=D;
        case 4 % Position 4 : Down/Left
    im_L=L;
    im_D=D;
    end
    ha = axes('units','normalized', ...
    'position',[im_L im_D  im_W im_H]);
    %%
    % Load in a background image and display it using the correct 
    I=imread(Image_file);
    hi = imagesc(I);
    % colormap gray
    %%
    % Turn the handlevisibility off so that we don't inadvertently plot into the axes again
    % Also, make the axes invisible
    set(ha,'handlevisibility','off','visible','off');
    
    % this creates transparency, you probably dont need it:
    % set(hi,'alphadata',.5)
    % % move the image to the top:
    uistack(ha,'bottom');
    %%
end
