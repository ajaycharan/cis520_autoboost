classdef FeatureSelector
    %FEATURESELECTOR Class for performing automatic feature selection.
    
    properties(Access=public)
        bns_scores = [];
        bns_max = 0;
        feat_ind = [];
        binary = false;
        scale = 'none';
    end
    
    methods
        function obj = FeatureSelector()
        end
        
        function Xout = apply(FS, X)
            % APPLY Apply a feature selector to a set of observations.
            %
            % Parameters:
            %   'FS'
            %       FeatureSelector to use for selection.
            %   'X'
            %       Matrix to apply feature selection on.
            %
            % Return values:
            %   'Xout'
            %       Output featurespace after selection and scaling.
            
            if (size(X,1)==1)
                % single row - use mex file
                Xout = fast_featureselect(FS, X);
                return;
            end
            
            if FS.binary
                % word presence only
                Xout = double(X > 0);
            else
                Xout = X;
            end

            % threshold
            if numel(FS.feat_ind) > 0
                Xout = Xout(:, FS.feat_ind);
            end
            
            % decompress scores
            scores = double(FS.bns_scores);
            scores = scores * (FS.bns_max * 0.0000152590219);
           
            switch FS.scale
                case 'bns'
                    if size(Xout,1)==1
                        Xout = Xout .* scores;
                    else
                        Xout = bsxfun(@times, Xout, scores);
                    end
                otherwise
                    % no scaling
            end
        end
    end
    
    methods(Static)
        function FS = train(X, Y, varargin)
            % TRAIN Train a new feature selector using X,Y
            %
            %   Options available through varargs:
            %
            %       'scale'
            %           'bns' or 'none'. Type of scaling to do.
            %       'thresh_bns'
            %           Threshold value for BNS feature culling.
            %       'binary'
            %           Set to true in order to use binary features.
            
            FS = FeatureSelector();
            
            defaults.scale = 'none';   %   'bns', or 'none'
            defaults.thresh_bns = 0;
            defaults.binary = false;

            % parse options
            options = propval(varargin, defaults);

            % check inputs
            valid_scale_modes = {'bns','none'};
            if ~any(strcmp(options.scale,valid_scale_modes))
                error('Invalid scaling mode selected');
            end
            
            % save only necessary indices
            bns = calc_bns(X,Y);
            indices = bns >= options.thresh_bns;
            indices = find( indices );              % numeric indices
            
            FS.feat_ind = uint32(indices);          % compress to 32 bit int      
            scores =  bns( indices );               % only used features
            
            % compress to unsigned 16 bit integer for storage
            FS.bns_max = max(scores);
            if (FS.bns_max > 0)
                scores = scores / FS.bns_max; % normalize
            end
            scores = floor(scores * 65535);
            FS.bns_scores = uint16(scores); % save
            
            FS.binary = options.binary;
            FS.scale = options.scale;                        
        end
    end
    
end

