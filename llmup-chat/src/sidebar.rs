use gpui::*;
use gpui_component::{
    ActiveTheme, IconName, Root, Side, TitleBar, blue_400,
    button::{Button, ButtonVariants},
    context_menu::ContextMenuExt,
    green_400,
    group_box::GroupBox,
    h_flex,
    input::{InputState, TextInput},
    label::Label,
    resizable::{ResizableState, h_resizable, resizable_panel},
    sidebar::{Sidebar, SidebarHeader, SidebarMenu, SidebarMenuItem},
    v_flex,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Item {
    Playground,
    Models,
    Settings,
}

impl Item {
    pub fn label(self) -> &'static str {
        match self {
            Self::Playground => "Playground",
            Self::Models => "Models",
            Self::Settings => "Settings",
        }
    }

    pub fn icon(self) -> IconName {
        match self {
            Self::Playground => IconName::SquareTerminal,
            Self::Models => IconName::Bot,
            Self::Settings => IconName::Settings2,
        }
    }

    pub fn handler(
        self,
    ) -> impl Fn(&mut AppSidebar, &ClickEvent, &mut Window, &mut Context<AppSidebar>) + 'static
    {
        move |app_sidebar, _, _, cx| {
            //
            app_sidebar.active = self;
            cx.notify()
        }
    }
}

pub struct AppSidebar {
    collapsed: bool,
    active: Item,
}

impl AppSidebar {
    pub fn new(_: &mut Window, cx: &mut Context<Self>) -> Self {
        Self {
            collapsed: false,
            active: Item::Playground,
        }
    }

    pub fn active(&self) -> Item {
        self.active
    }
}

impl Render for AppSidebar {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let group = vec![Item::Playground, Item::Models, Item::Settings];

        Sidebar::new(Side::Left)
            .collapsed(self.collapsed)
            .header(SidebarHeader::new())
            .child(SidebarMenu::new().children(group.iter().map(|item| {
                SidebarMenuItem::new(item.label())
                    .icon(item.icon())
                    .on_click(cx.listener(item.handler()))
            })))
    }
}
