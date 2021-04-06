import torch
import torch.nn as nn

from models.modules import build_mlp, SoftAttention, PositionalEncoding, ScaledDotProductAttention, create_mask, proj_masking, PositionalEncoding

class VisualSoftDotAttention(nn.Module):
    ''' Visual Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):#v_dim : changed: 720 * 2203
        '''Initialize layer.'''
        #print('h_dim',h_dim)#512
        #print('v_dim',v_dim)#2176
        super(VisualSoftDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)#(input : 512, output : 256)
        self.linear_in_v = nn.Linear(v_dim, dot_dim, bias=True)# input : 2176, output : 256
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, visual_context, mask=None):#
        '''Propagate h through the network.

        h: batch x h_dim #h: [10,512]
        visual_context: batch x v_num x v_dim #([10, 36, 2176])
        '''
        visual_context = visual_context.cuda(async=True)
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1 #[10,256,1]
        context = self.linear_in_v(visual_context)  # batch x v_num x dot_dim #[10,36,256]

        # Get attention
        #bmm : (10,36,256) * (10,256,1) ==>( 10,36, 1)
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num
        #print('attn',attn.shape)#10,36
        attn = self.sm(attn)
        #print('attn_2',attn.shape)#10,36
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num
        #print('attn3',attn3.shape)#10,1,36

        weighted_context = torch.bmm(
            attn3, visual_context).squeeze(1)  # batch x v_dim

        #weighted_context.shape)#10,2176
        return weighted_context, attn

class ObjectDotAttention(nn.Module):
    ''' Object Dot Attention Layer. '''

    def __init__(self, h_dim, v_dim, dot_dim=256):#v_dim : changed: 720 * 2203#256
        '''Initialize layer.'''

        super(ObjectDotAttention, self).__init__()
        self.linear_in_h = nn.Linear(h_dim, dot_dim, bias=True)#(input : 512, output : 256)
        self.linear_in_o = nn.Linear(v_dim, dot_dim, bias=True)# input : 2176, output : 256
        self.sm = nn.Softmax(dim=1)

    def forward(self, h, object_context, mask=None):#
        '''Propagate h through the network.

        h: batch x h_dim #h: [BatchSize,256]
        object_context: batch x v_num x v_dim #([BatchSize, 12, 17])
        '''

        object_context= object_context.cuda(async=True)
        #print('h',h.shape)#torch.Size([4, 256])
        #print('object_context',object_context.shape) #torch.Size([4, 12, 17])
        target = self.linear_in_h(h).unsqueeze(2)  # batch x dot_dim x 1 # 10 256 1
        #print('target', target.shape)#torch.Size([4, 256, 1])
        context = self.linear_in_o(object_context)

        #print('context', context.shape)

        # Get attention
        #bmm : (10,36,256) * (10,256,1) ==>(10,36,1) ex. (b,n,m) * (b,m,p) = (b,n,p)
        attn = torch.bmm(context, target).squeeze(2)  # batch x v_num => dimension size = 2 ==> delete
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x v_num #change the tensor size

        # attention metrics * context
        '''
        weighted_context = torch.bmm(
            attn3, object_context).squeeze(1)  # batch x v_dim
        '''
        weighted_context = torch.bmm(attn3, object_context).squeeze(1)  # batch x v_dim
        #print('weighted_context', weighted_context.shape, weighted_context) #10,1,17

        #weighted_context.shape) #10,2176
        return weighted_context, attn

class SelfMonitoring(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(SelfMonitoring, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)
        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size)
        self.lstm = nn.LSTMCell(2587, rnn_hidden_size)

        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])
        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        if opts.monitor_sigmoid:#NOT HERE
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:#THIS
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.num_predefined_action = 1
        self.object_t_size = 17  # object_size
        self.place_t_size = 10  # place_Size

        self.object_attention_layer = ObjectDotAttention(rnn_hidden_size, self.object_t_size)
        self.place_attention_layer = VisualSoftDotAttention(rnn_hidden_size, self.place_t_size)

    def forward(self, img_feat, navigable_feat, pre_feat, question, object_t, place_t, h_0, c_0, ctx, pre_ctx_attend,
                navigable_index=None, ctx_mask=None):
        """ Takes a single step in the decoder

        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size

        pre_feat: previous attended feature, batch x feature_size

        question: this should be a single vector representing instruction

        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """

        object_t = object_t.float()
        place_t = place_t.float()

        #print('h_0',h_0.shape)#([4, 256])
        #print('object_t',object_t.shape)#torch.Size([4, 12, 17])

        object_att, alpha_o = self.object_attention_layer(h_0, object_t)  # 10, 17
        place_att, alpha_p = self.place_attention_layer(h_0, place_t)  # 10, 10

        #print('object_att,place_att',object_att.shape,place_att.shape)#torch.Size([4, 1, 17]) torch.Size([4, 10])
        concat_o_p = torch.cat((object_att, place_att), 1)
        #print('concat_o_p', concat_o_p.shape)  # 10,12,27

        batch_size, num_imgs, feat_dim = img_feat.size()

        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat) # I think this is previous action
        positioned_ctx = self.lang_position(ctx)

        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)

        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx, concat_o_p), 1)
        #print('concat_input',concat_input.shape)#Before : 4, 512 / with concat_o_p : 4, 539

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        # value estimation
        concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))

        h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1)) # h_1_value is same with h_t_pm

        value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))#THIS IS PROGRESS MONITOR(Value is p_t_pm

        return h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, value, navigable_mask

class SpeakerFollowerBaseline(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(SpeakerFollowerBaseline, self).__init__()

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size

        self.proj_img_mlp = nn.Linear(img_feat_input_dim, img_fc_dim[-1], bias=fc_bias)

        self.proj_navigable_mlp = nn.Linear(img_feat_input_dim, img_fc_dim[-1], bias=fc_bias)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=False)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_feat_input_dim * 2, rnn_hidden_size)

        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=False)

        self.proj_out = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)

    def forward(self, img_feat, navigable_feat, pre_feat, h_0, c_0, ctx, navigable_index=None, ctx_mask=None):
        """ Takes a single step in the decoder LSTM.

        img_feat: batch x 36 x feature_sizedd
        navigable_feat: batch x max_navigable x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        # add 1 because the navigable index yet count in "stay" location
        # but navigable feature does include the "stay" location at [:,0,:]
        index_length = [len(_index)+1 for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_img_feat = proj_masking(img_feat, self.proj_img_mlp)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)

        weighted_img_feat, _ = self.soft_attn(self.h0_fc(h_0), proj_img_feat, img_feat)

        concat_input = torch.cat((pre_feat, weighted_img_feat), 1)

        h_1, c_1 = self.lstm(self.dropout(concat_input), (h_0, c_0))

        h_1_drop = self.dropout(h_1)

        # use attention on language instruction
        weighted_context, ctx_attn = self.soft_attn(self.h1_fc(h_1_drop), self.dropout(ctx), mask=ctx_mask)
        h_tilde = self.proj_out(weighted_context)

        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        return h_1, c_1, ctx_attn, logit, navigable_mask
